import copy, sys, os
import torch.distributed as dist
import torch
from torch import nn
import torch.nn.functional as F
import ot
import otfusion.ground_metric as gm
import otfusion.utils as utils
import otfusion.wasserstein_ensemble as we
ROOT = '/public/data0/NLP/zhangzheng15/otfusion'

def get_args(**new_args):
    default_args = {
        "model_name": "mlpnet",
        "gpu_id": -1,
        "disable_bias": True,
        # "dist_normalize": False,
        "geom_ensemble_type": "acts",
        "normalize_acts": "no", # get_activation
        "act_num_samples": 200,
        "activation_seed": 0,
        "activation_mode": "raw",
        "debug": False,
        "width_ratio": 1,
        "prelu_acts": True,
        "pool_acts": False, "pool_relu": False,
    }
    data_args = {
        "dataset": "mnist",
        "batch_size_train": 256,
        "batch_size_test": 256,
        "to_download": True,
        "skip_idxs": [],
        "personal_class_idx": 9,
        "seed": 1,
    }
    importance_args = {
        "importance": "uniform",
        "importance_method": "uniform",  # none
        "softmax_temperature": 10 # importance==acts
    }
    ground_metric_args = {
        "clip_gm": False,
        "ground_metric_normalize": "none",
        "ground_metric": "euclidean",
        "normalize_coords": False,  # mectric
        "dtype": torch.float64,
        "mem_efficient": False
    }
    ot_args = {
        "gromov": False,
        "reg": 0.001,
        "proper_marginals": True,
        "exact": True,
    }
    retrain_args = {
        "rt_lr": 0.01,
        "rt_lr_decay": 1,
        "rt_seed": 1,
        "rt_momentum": 0.9,
        "rt_epoch": 40,
        "retrain_lr_decay_factor": 2,
        "retrain_lr_decay_epochs": "15_15_10",
        "rt_log_interval": 100,
        "rt_result_dir" : '/workspace/zhangzheng/otfusion/checkpoints',
        "rt_exp_name": 'exp_name',
        'rt_model_save': False,
        "rt_batch_size": 128
    }
    args_dict={   
        **default_args,
        **data_args,
        **importance_args,
        **ground_metric_args,
        **ot_args,
        **retrain_args,
        # "num_hidden_nodes1": 400,
        # "num_hidden_nodes2": 200,
        # "num_hidden_nodes3": 100,
        **new_args
    }

    class X:
        def __init__(self, args_dict) -> None:
            for k,v in args_dict.items():
                self.__dict__[k]= v
    return X(args_dict)

def normalize_tensor(tens):
    tens_shape = tens.shape
    assert tens_shape[1] == 1
    tens = tens.view(tens_shape[0], 1, -1)
    norms = tens.norm(dim=-1)
    ntens = tens/norms.view(-1, 1, 1)
    ntens = ntens.view(tens_shape)
    return ntens

def regist_hooks(module, parent_name, get_activation_hook, activations, hook_handles):
    for name, submodule in module.named_children():
        name_ = parent_name + '.' + name if parent_name else name
        if hasattr(submodule, 'weight') or hasattr(submodule, 'bias') or len(list(module.children()))==0:
            # print("set forward hook for module named: ", name_)
            # activations[name_] = {'input': [], 'output': []}
            hook = get_activation_hook(activations, name_)
            hook_handles.append(submodule.register_forward_hook(hook))
            if len(list(module.children())) > 0:
                regist_hooks(submodule, name_, get_activation_hook, activations, hook_handles)
        else:
            # print("ignore forward hook for module named: ", name_)
            regist_hooks(submodule, name_, get_activation_hook, activations, hook_handles)

def normalize_activations(args, activations, mode):
    for layer in activations.keys():
        # 将数组重新拼成Tensor
        activations[layer] = torch.stack(activations[layer]).squeeze(1)
        # print("min of act of layer {}: {}, max: {}, mean: {}".format(layer, torch.min(activations[layer]), torch.max(activations[layer]), torch.mean(activations[layer])))

        if mode == 'mean':
            activations[layer] = activations[layer].mean(dim=0)
        elif mode == 'std':
            activations[layer] = activations[layer].std(dim=0)
        elif mode == 'meanstd':
            activations[layer] = activations[layer].mean(dim=0) * activations[layer].std(dim=0)

        if args.normalize_acts == "stand":
            print("standarize_acts")
            mean_acts = activations[layer].mean(dim=0)
            std_acts = activations[layer].std(dim=0)
            activations[layer] = (activations[layer] - mean_acts)/(std_acts + 1e-9)
        elif args.normalize_acts == "center":
            print("center_acts")
            mean_acts = activations[layer].mean(dim=0)
            activations[layer] = (activations[layer] - mean_acts)
        elif args.normalize_acts == "norm":
            print("normalizing the activation vectors")
            activations[layer] = normalize_tensor(activations[layer])
            print("min of act: {}, max: {}, mean: {}".format(torch.min(activations[layer]),
                                                                torch.max(activations[layer]),
                                                                torch.mean(activations[layer])))

        print("activations at layer {} have the following shape ".format(layer), activations[layer].shape)

def get_inputs_for_acts(args, train_loader):
    num_samples_processed = 0
    num_personal_idx = 0
    targets = []
    targets_hist = {}
    prepared_data = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if num_samples_processed == args.act_num_samples:
            break
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
        

        if args.skip_idxs and int(target.item()) in args.skip_idxs:
            continue

        if int(target.item()) == args.personal_class_idx:
            num_personal_idx += 1    

        prepared_data.append((data, target))
        targets_hist[target.item()] = targets_hist.get(target.item(), 0)+1
        # targets.append(target.item())

        num_samples_processed += 1
    print(">>> statistics about the data used to generate activations:")
    print("skip_idxs:", args.skip_idxs)
    print("personal_class_idx:", args.personal_class_idx, "; num_personal_idx:", num_personal_idx)
    print("targets_hist:", targets_hist)
    return prepared_data#, targets


def compute_activations_across_models_zz(args, models, inputs, dump_activations=False, dump_path=None, activations=None):
    def get_activation_hook(activation, name):
        def hook(model, input, output):
            if name not in activation:
                activation[name] = []
            activation[name].append(output.detach())
        return hook

    torch.manual_seed(args.activation_seed)
    
    activations = {} if activations == None else activations

    hook_handles = []
    assert args.disable_bias

    # handle below for bias later on!
    # Set forward hooks for all layers inside a model
    for idx, model in enumerate(models):
        activations[idx] = {}
        regist_hooks(model, "", get_activation_hook, activations[idx], hook_handles)
    
    for data, target in inputs:
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
        for idx, model in enumerate(models):
            model(data)


    # Combine the activations generated across the number of samples to form importance scores
    # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'

    for idx, model in enumerate(models):
        normalize_activations(args, activations[idx], args.activation_mode)   
        # Dump the activations for all models onto disk
        if dump_activations and dump_path is not None:
            utils.save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for hook_handle in hook_handles:
        # print(type(hook_handle))  # <class 'torch.utils.hooks.RemovableHandle'>
        hook_handle.remove()
    return activations


def align_layer(args, layer0_weight, layer1_weight, activation0: torch.Tensor, activation1, is_conv=False, feature_dim=1):
    # 这里需要squeeze 一次，因为activation的shape 是 batch,1,feature 之所以有这个1，应该是前面获取activations 留下的问题
    
    # 把当前层feature维放到最前面(原文是先把batch维放到了最后), 
    # 这个做法更慢，并且两种算法目前都后续过程都没有影响 
    activation0 = torch.einsum('bf...->fb...', activation0)
    activation1 = torch.einsum('bf...->fb...', activation1)
    ## 原做法
    # activation0 = activation0.permute([*range(1, len(activation0.shape)), 0]).contiguous()
    # activation1 = activation1.permute([*range(1, len(activation1.shape)), 0]).contiguous()
    activation0 = activation0.reshape(activation0.shape[0], -1)
    activation1 = activation1.reshape(activation0.shape[0], -1)
    print("activation congigous:", activation0.is_contiguous(), activation0.is_contiguous())

    print("activation0.shape:", activation0.shape)
    print("activation1.shape:", activation1.shape)

    assert activation0.shape[0] == layer0_weight.shape[0]  # make sure feature dim align
    assert activation1.shape[0] == layer1_weight.shape[0]

    if args.geom_ensemble_type == 'acts':
        # M0, M1 = we._process_ground_metric_from_acts(args, is_conv, ground_metric_object,
                                                        #   [activation0, activation1])
        M0 = gm.GroundMetric.PROCESS(args, activation0, activation1)
    elif args.geom_ensemble_type == "wts":
        coordinates0 = layer0_weight.reshape(layer0_weight.shape[0], -1)
        coordinates1 = layer1_weight.reshape(layer1_weight.shape[0], -1)

        M0 = gm.GroundMetric.PROCESS(args, coordinates0, coordinates1)

    # print(args.dist_normalize)
    # print(args.activation_histograms)
    ## todo 
    # print("M", M0)
    mu_cardinality = layer0_weight.shape[0]
    nu_cardinality = layer1_weight.shape[0]
    
    mu = utils.get_histogram(args, mu_cardinality, activation=activation0, layer_weight=layer0_weight, dtype=args.dtype)
    nu = utils.get_histogram(args, nu_cardinality, activation=activation1, layer_weight=layer1_weight, dtype=args.dtype)
    print(f"(max, min) of mu, nu: ({mu.min().item():.5f}, {mu.max().item():.5f}), ({nu.min().item():.5f}, {nu.max().item():.5f})",
            ";\tsum of mu,nu:", mu.sum().item(), nu.sum().item())
    cpuM = M0.data.detach().cpu().type(args.dtype)  #.numpy()
    print("args.exact:", args.exact)
    if args.exact:
        # shape of T (mu_cardinality, nu_cardinality)
        T = ot.emd(mu, nu, cpuM)
    else:
        T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)

    ot_cost = torch.multiply(T, cpuM).sum()
    print("distance:", ot_cost.item())

    T_var,_ = we._compute_marginals(args, torch.tensor(T))
    print("error value of T:", (T_var.sum(axis=1).squeeze() - torch.ones(T_var.shape[0])).sum())
    print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
    print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
    print("dtype of T, mu, nu, layer0_weight_shape: ", T_var.dtype, mu.dtype, nu.dtype, layer0_weight.dtype)
    # print("args.dtype: ", args.dtype)
    layer0_weight_shape = layer0_weight.shape
    aligned_layer0_weight = torch.matmul(T_var.t(), layer0_weight.view(layer0_weight_shape[0], -1).type(args.dtype)).float()
    aligned_layer0_weight = aligned_layer0_weight.view(layer0_weight_shape)

    return aligned_layer0_weight, T_var, mu, nu


def fuse_model(args, model, model1, activations, inplace=True, recompute_acts=False, acts_inputs=None):
    """
    discard
    """
    model0 = model if inplace else copy.deepcopy(model)

    T_var = None
    aligned_params = {}
    avg_aligned_params = {}
    with torch.no_grad():
        for (layer0_name, layer0_weight), (layer1_name, layer1_weight) in zip(model0.named_parameters(), model1.named_parameters()):
            is_conv = len(layer0_weight.shape) != 2
            activation0 = activations[0][layer0_name.rstrip(".weight")]
            activation1 = activations[1][layer1_name.rstrip(".weight")]

            print("\n\n>>> layer name:", layer0_name, layer1_name)
            print("layer shape:", layer0_weight.shape, layer1_weight.shape)
            print("activation shape:", activation0.shape, activation1.shape)
            if T_var != None:
                print("T_var.shape:", T_var.shape)
            
            layer0_aliged = layer0_weight.clone()
            if T_var != None:
                layer0_weight_shape = layer0_weight.shape
                if is_conv:
                    print("zzzz tmp: ", layer0_weight.shape, T_var.shape)
                    layer0_aliged = torch.einsum("oiwh,ix->oxwh",layer0_weight.type(args.dtype), T_var)
                else:
                    layer0_aliged = layer0_weight.type(args.dtype) @ T_var
                layer0_aliged = layer0_aliged.view(layer0_weight_shape).contiguous()

                if recompute_acts and acts_inputs != None:
                    activations[0] = compute_activations_across_models_zz(args, [model0], acts_inputs, activations)[0]
                    activation0_update = activations[0][layer0_name.rstrip(".weight")]
                    print("zzzz-D difference caused by re computations", (activation0-activation0_update).abs().sum())
            
            layer0_aliged, T_var, _, _ = align_layer(args, layer0_aliged, layer1_weight, activation0, activation1)
            avg_aligned_params[layer0_name] = (layer0_aliged+layer1_weight)/2
            aligned_params[layer0_name] = layer0_aliged
            utils.update_model(model0, avg_aligned_params)
    aligned_model0 = copy.deepcopy(model0)
    utils.update_model(aligned_model0, aligned_params)
    # geometric_acc, geometric_model = we.get_network_from_param_list(args, avg_aligned_params, test_loader)
    return model0, aligned_model0

def fuse_resnet_model(args, model, model1, activations, inplace=True, recompute_acts=False, acts_inputs=None):
    """
    discard
    """
    model0 = model if inplace else copy.deepcopy(model)
    T_var = None
    aligned_params = {}
    avg_aligned_params = {}

    skip_T_var = None
    skip_T_var_idx = -1
    residual_T_var = None
    residual_T_var_idx = -1
    with torch.no_grad():
        for idx, ((layer0_name, layer0_weight), (layer1_name, layer1_weight)) in enumerate(zip(model0.named_parameters(), model1.named_parameters())):
            is_conv = len(layer0_weight.shape) != 2
            activation0 = activations[0][layer0_name.rstrip(".weight")]
            activation1 = activations[1][layer1_name.rstrip(".weight")]

            print("\n\n>>> layer name:", layer0_name, layer1_name)
            print("layer shape:", layer0_weight.shape, layer1_weight.shape)
            print("activation shape:", activation0.shape, activation1.shape)
            if T_var != None:
                print("T_var.shape:", T_var.shape)
            
            layer0_aliged = layer0_weight.clone()
            if T_var != None:
                layer0_weight_shape = layer0_weight.shape
                if is_conv:
                    assert len(layer0_weight_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer0_weight_shape[1] != layer0_weight_shape[0]:
                        if not (layer0_weight_shape[2] == 1 and layer0_weight_shape[3] == 1):
                            print(f'saved skip T_var at layer {idx} with shape {layer0_weight_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            print(
                                f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        print("shape of previous transport map now is", T_var.shape)
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        else:
                            print("doing nothing for skips")

                    print("zzzz tmp: ", layer0_weight.shape, T_var.shape)
                    layer0_aliged = torch.einsum("oiwh,ix->oxwh",layer0_weight.type(args.dtype), T_var)
                else:
                    layer0_aliged = layer0_weight.type(args.dtype) @ T_var
                layer0_aliged = layer0_aliged.view(layer0_weight_shape).contiguous()

                if recompute_acts and acts_inputs != None:
                    activations[0] = compute_activations_across_models_zz(args, [model0], acts_inputs, activations)[0]
                    activation0_update = activations[0][layer0_name.rstrip(".weight")]
                    print("zzzz-D difference caused by re computations", (activation0-activation0_update).abs().sum())
            
            layer0_aliged, T_var, _, _ = align_layer(args, layer0_aliged, layer1_weight, activation0, activation1)
            avg_aligned_params[layer0_name] = (layer0_aliged+layer1_weight)/2
            aligned_params[layer0_name] = layer0_aliged
            utils.update_model(model0, avg_aligned_params)

    aligned_model0 = copy.deepcopy(model0)
    utils.update_model(model0, aligned_params)
    # geometric_acc, geometric_model = we.get_network_from_param_list(args, avg_aligned_params, test_loader)
    return model0, aligned_model0

def get_fisher_info(model, train_loader, num_samples, batch_size=1, grad_type="square"):
    fisher_infos = dict(model.named_parameters())
    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square
    
    for name, p in fisher_infos.items():
        fisher_infos[name] = torch.zeros_like(p)
        # p = torch.zeros_like(p)
        # print(fisher_infos[name])
    criterion = nn.CrossEntropyLoss()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        if i * batch_size >= num_samples:
            break
        # print(input.shape, target)
        output = model(input)
        loss = criterion(output, target)
        # print(output.argmax(), target)
        loss.backward()
        for name, p in model.named_parameters():
            fisher_infos[name] += grad_method(p.grad).data
            # print(p.grad)
            # break
            # p.grad.zero_()
        model.zero_grad()
        print(fisher_infos["fc1.weight"].sum())
        # break
    for name, p in fisher_infos.items():
        fisher_infos[name] = fisher_infos[name]*100/ fisher_infos[name].sum()
    
    return fisher_infos

def avg_model(model0, model1, weight0={}, weight1={}, model0_weight=1, model1_weight=1, eps=1e-11, test_loader=None, args=None, feature_dim_=None):
    """
    args: weight0, weight1 : int/float/dict of tensor(compitable with model.named_parameters)
    """
    avged_parameters = {}
    model0_parameters = dict(model0.named_parameters())
    model1_parameters = dict(model1.named_parameters())
    model0_weight *= 1e7
    model1_weight *= 1e7

    for name, p in model0_parameters.items():
        if weight0 and weight1 and ( weight0[name].shape != model0_parameters[name].shape):
            if feature_dim_ == None:
                feature_dim = 0
            elif type(feature_dim_) == int:
                feature_dim = feature_dim_
            else:
                feature_dim = feature_dim_(name)
            cardinality = p.size(feature_dim)
            weight0.setdefault(name, torch.ones((cardinality)))
            weight1.setdefault(name, torch.ones((cardinality)))

            # print("zzzz avg_model")
            shape = torch.ones((len(model0_parameters[name].shape)), dtype=torch.int)
            shape[feature_dim] = cardinality
            shape = shape.tolist()
            weight0[name] = weight0[name].reshape(shape)
            weight1[name] = weight1[name].reshape(shape)
        if weight0:
            param_weight0 = weight0[name]
            param_weight1 = weight1[name]
        else:
            param_weight0 = 1
            param_weight1 = 1

        p0_mask = (model0_parameters[name].abs() < 1e-7)
        p1_mask = (model1_parameters[name].abs() < 1e-7)
        # print((p0_mask & p1_mask).sum(), p0_mask.numel())
        # print(weight0[name])
        # print((model0_parameters[name] * weight0[name] * model1_weight).max())
        avged_parameters[name] = (model0_parameters[name] * param_weight0 * model0_weight + model1_parameters[name] * param_weight1 * model1_weight)  \
            / (param_weight0*model0_weight + param_weight1*model1_weight + eps)
        avged_parameters[name][p0_mask & p1_mask] = 0

    # print(model0_weight, model1_weight)
    avged_model = copy.deepcopy(model0)
    utils.update_model(avged_model, avged_parameters)
    if test_loader != None:
        assert args != None
        utils.test(args, avged_model, test_loader)
    return avged_model

def prune_model(model0, thred):
    model = copy.deepcopy(model0)
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        p[p.abs()<thred] = 0
    return model

def regist_backward_hooks(module, parent_name, get_activation_hook, activations, hook_handles):
    for name, submodule in module.named_children():
        name_ = parent_name + '.' + name if parent_name else name
        if hasattr(submodule, 'weight') or len(list(module.children()))==0:
            # print("set backward hook for module named: ", name_)
            hook = get_activation_hook(activations, name_)
            hook_handles.append(submodule.register_backward_hook(hook))
        else:
            # print("ignore backward hook for module named: ", name_)
            regist_hooks(submodule, name_, get_activation_hook, activations, hook_handles)

def compute_activations_gredients_across_models(args, models, inputs, activation_gradients=None, grad_type="square"):
    def get_backward_hook(gradients, name):
        def hook(module, grad_in, grad_out):
            if name not in gradients:
                gradients[name] = []

            if type(grad_out) == tuple:
                grad_out = grad_out[0]
                # assert len(grad_out) == 1

            gradients[name].append(grad_out.detach())
        return hook

    torch.manual_seed(args.activation_seed)
    
    activation_gradients = {} if activation_gradients == None else activation_gradients

    hook_handles = []
    assert args.disable_bias

    # handle below for bias later on!
    # Set forward hooks for all layers inside a model
    for idx, model in enumerate(models):
        activation_gradients[idx] = {}
        regist_backward_hooks(model, "", get_backward_hook, activation_gradients[idx], hook_handles)
    


    criterion = nn.CrossEntropyLoss()
    for model in models:
        model.train()
        for i, data in enumerate(inputs):
            if type(data) == tuple:
                input, target = data
                # print(input.shape, target)
                output = model(input)
                # loss = criterion(output, target)
                loss = F.nll_loss(output, target, size_average=False)
            elif type(data) == dict:
                outputs = model(**data)
                loss = outputs["loss"]

            # print(output.argmax(), target)
            loss.backward()
            # for name, p in model.named_parameters():
            #     fisher_infos[name] += grad_method(p.grad).data
                # print(p.grad)
                # break
                # p.grad.zero_()
            model.zero_grad()

    ## fisher information
    fisher_infos = {}
    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square
    


    for idx in activation_gradients.keys():
        fisher_infos[idx] = {}
        for name in activation_gradients[idx].keys():
            activation_gradients[idx][name] = torch.cat(activation_gradients[idx][name], dim=0)            
            # print(activation_gradients[idx][f"{name}"].shape)
            
            axis = [0] + list(range(2, len(activation_gradients[idx][name].shape)))
            # print(axis)
            fisher_infos[idx][f"{name}.weight"] = grad_method(activation_gradients[idx][name]).sum(axis=axis)
            # print(fisher_infos[idx][f"{name}.weight"].shape)

        # for name, p in fisher_infos[idx].items():
        #     fisher_infos[idx][name] = fisher_infos[idx][name]*100/ fisher_infos[idx][name].sum()
        

    # Combine the activations generated across the number of samples to form importance scores
    # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'

    # for idx, model in enumerate(models):
    #     normalize_activations(args, activations[idx], args.activation_mode)   
    #     # Dump the activations for all models onto disk
    #     if dump_activations and dump_path is not None:
    #         utils.save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for hook_handle in hook_handles:
        # print(type(hook_handle))  # <class 'torch.utils.hooks.RemovableHandle'>
        hook_handle.remove()
    return activation_gradients, fisher_infos

def fisher_avg_model(args, model0, model1, acts_inputs, test_loader=None,  test=False):
    activation_gradients1, fisher_infos = compute_activations_gredients_across_models(args, [model0, model1], acts_inputs)

    fisher_avged_model = avg_model(model0, model1, fisher_infos[0], fisher_infos[1])
    if test:
        _=utils.test(args, fisher_avged_model, test_loader)


def get_wrong_samples(args, model0, model1, train_loader):
    model0_wrong = utils.test(args, model0, train_loader)
    model1_wrong = utils.test(args, model1, train_loader)
    # len(model1_wrong), len(model0_wrong)
    # import copy
    wrong_samples = copy.deepcopy(model0_wrong)
    wrong_samples.extend(model1_wrong)
    print(f"number of wrong samples: {len(wrong_samples)}={len(model0_wrong)}+{len(model1_wrong)}")
    return wrong_samples

def dist_init(port=28550, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend, init_method="env://", rank=0, world_size=1)

def load_head(classifier, pretrained_model_path):
    print("loading parameters of classifier")
    pretrained_state_dict = torch.load(pretrained_model_path)
    pretrained_cls_state_dict_keys = [k for k in  pretrained_state_dict.keys() if k.startswith('cls')]
    
    old_keys = pretrained_cls_state_dict_keys
    new_keys = [key.lstrip("cls.") for key in pretrained_cls_state_dict_keys]
    for i in range(len(pretrained_cls_state_dict_keys)):
        new_key = new_keys[i]
        old_key = old_keys[i]
        if 'gamma' in new_key:
            new_keys[i] = new_key.replace('gamma', 'weight')
        if 'beta' in new_key:
            new_keys[i] = new_key.replace('beta', 'bias')
        if 'eq_relationship' in new_key:
            new_keys[i] = new_key.replace('eq_relationship', 'seq_relationship')

    pretrained_state_dict2 = {new_keys[i]: pretrained_state_dict[old_keys[i]] for i in range(len(new_keys))}
    print(pretrained_state_dict2.keys())
    print(dict(classifier.named_parameters()).keys())
    classifier.load_state_dict(pretrained_state_dict2, strict=False)

def bert_prepare(head="mlm"):
    import utils_zz2 as uz2
    from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    import transformers
    mlp_args = {
        'dataset': 'Cifar10',
        'model_name': 'resnet18_nobias_nobn'
    }
    args = get_args(**mlp_args)

    MODEL_PATH = "/public/data0/NLP/zhangzheng15/datasets/pretrained/bert-large-uncased"
    model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
    # # 修改配置
    # model_config.output_hidden_states = True
    # model_config.output_attentions = True
    # # print(">>> load huggingface model")
    if head == None:
        hf_model = transformers.BertModel.from_pretrained(MODEL_PATH, config=model_config)
    elif head == "mlm":
        hf_model = transformers.BertForMaskedLM.from_pretrained(MODEL_PATH, config=model_config)
        # head = BertOnlyMLMHead(model_config)
        # load_head(head, f"{MODEL_PATH}/pytorch_model.bin")

    tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_PATH, config=model_config)

    
    sys.path.append('/workspace/zhangzheng/bing_bert')
    import v2.model_load as ml
    data_args = ml.construct_arguments()
    batch, dict_input, dataset_iterator = ml.load_batch(data_args, 64)
    sys.path.remove('/workspace/zhangzheng/bing_bert')

    acts_inputs = [{'input_ids': dict_input['input_ids']}]
    activations = uz2.compute_activations_across_models_zz(args, [hf_model], acts_inputs)

    return args, [hf_model], model_config, tokenizer, activations, acts_inputs

def test_mlm_bert(model, tokenizer, head=None, text="Hello I'm a [MASK] model."):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    # from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    # head = BertOnlyMLMHead(model_config)
    if head == None:
        prediction_scores = output['logits']
    else:
        prediction_scores = head(output[0])
    print(tokenizer.decode(prediction_scores.squeeze().argmax(axis=1)))
    # encoded_input
    # encoded_input['input_ids'], prediction_scores.squeeze().argmax(axis=1)

