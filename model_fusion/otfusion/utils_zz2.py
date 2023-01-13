from typing import OrderedDict
import torch
import ot
import otfusion.utils as utils
import otfusion.utils_zz as uz
import otfusion.ground_metric as gm
import otfusion.activation_utils as au
import copy


def print_list_tensor_shape(l):
    for t in l:
        print(t.shape, end=',')
    print()


def compute_activations_across_models_zz(args, models, inputs, dump_activations=False, dump_path=None, activations=None):

    def get_activation_hook(activation, name):

        def hook(model, inputs, output):
            if name not in activation:
                activation[name] = {'input': [], 'output': []}
            if type(inputs)== tuple:
                inputs = [inp.detach() for inp in inputs]
            
            assert len(inputs) == 1
            activation[name]['input'].append(inputs[0])
            activation[name]['output'].append(output.detach())

        return hook

    torch.manual_seed(args.activation_seed)
    activations = {} if activations == None else activations
    hook_handles = []
    assert args.disable_bias

    for idx, model in enumerate(models):
        activations[idx] = OrderedDict()
        uz.regist_hooks(model, "", get_activation_hook, activations[idx], hook_handles)

    if not type(inputs) == list:
        inputs = [inputs]
    for data in inputs:
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
        for idx, model in enumerate(models):
            if type(data) == tuple:
                model(data[0])
            elif type(data) == dict:
                model(**data) 

    for hook_handle in hook_handles:
        hook_handle.remove()
    return activations


def align_activation(args, activation0: torch.Tensor, activation1, feature_dim=-1, eps=1e-5):
    def check_dim_identify(tensors, dim=-1):
        a = tensors[0].shape[dim]
        for tensor in tensors:
            if tensor.shape[dim] != a:
                return False
        return True

    # if type(activation0) == list:
    try:
        if type(activation0) == list:
            if len(activation0) == 1:
                activation0 = activation0[0]
            else:
                activation0 = torch.cat(activation0, dim=0)
        elif type(activation0) == torch.Tensor:
            pass
        
        if type(activation1) == list:
            if len(activation1) == 1:
                activation1 = activation1[0]
            else:        
                activation1 = torch.cat(activation1, dim=0)
        elif type(activation1) == torch.Tensor:
            pass
    except Exception as e:
        print(e)
        # print_list_tensor_shape(activation0)
        # if check_dim_identify(activation0, dim=-1):
        #     activation0 = [a.reshape(-1, a.shape[-1]) for a in activation0]
        #     activation1 = [a.reshape(-1, a.shape[-1]) for a in activation1]
        #     activation0 = torch.cat(activation0, dim=0)
        #     activation1 = torch.cat(activation1, dim=0)
        # else:
        return None, None, None
    # 这里需要squeeze 一次，因为activation的shape 是 batch,1,feature 之所以有这个1，应该是前面获取activations 留下的问题

    M0, dis = p_dis(activation0, activation1, args, feature_dim)    

    M0 = M0 + (torch.eye(M0.size(0)) == 0) * eps
    

    # print(args.dist_normalize)
    # print(args.activation_histograms)
    ## todo 
    # print("M", M0)
    mu_cardinality = activation0.shape[feature_dim]
    nu_cardinality = activation1.shape[feature_dim]
    
    mu = utils.get_histogram(args, mu_cardinality, activation=activation0, dtype=args.dtype)
    nu = utils.get_histogram(args, nu_cardinality, activation=activation1, dtype=args.dtype)
    mu *= mu_cardinality
    nu *= mu_cardinality

    # print(f"(max, min) of mu, nu: ({mu.min().item():.5f}, {mu.max().item():.5f}), ({nu.min().item():.5f}, {nu.max().item():.5f})",
    #         ";\tsum of mu,nu:", mu.sum().item(), nu.sum().item())
    cpuM = M0.data.detach().cpu().type(args.dtype)

    # print(type(mu), type(nu), type(cpuM))
    # print("args.exact:", args.exact)
    if args.exact:
        # shape of T (mu_cardinality, nu_cardinality)
        T = ot.emd(mu.numpy(), nu.numpy(), cpuM.numpy())
    else:
        T = ot.bregman.sinkhorn(mu.numpy(), nu.numpy(), cpuM, reg=args.reg)

    T = torch.tensor(T)
    ot_cost = torch.multiply(T, cpuM).sum()

    # T_var,_ = we._compute_marginals(args, T)
    # print("error value of T:", (T_var.sum(axis=1).squeeze() - torch.ones(T_var.shape[0])).sum())
    print(f"Ratio of trace to the matrix sum:  {(torch.trace(T)/torch.sum(T)).item():.4f};\t otcost: , {ot_cost.item():.2f};\t distance:{dis}")
    
    # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))

    return T, mu, nu

def align_activations(args, activations):
    activations_T = dict(zip(activations[0].keys(), [None]*len(activations[0].keys())))
    for name in activations[0].keys():
        print(f">>> {name}")
        is_conv = (activations[0][name]['input'].ndim == 4)
        if is_conv:
            feature_dim = 1
        else:
            feature_dim = -1
        T_var_in, _, _ = align_activation(args, activations[0][name]['input'], activations[1][name]['input'], feature_dim)
        T_var_out, _, _ = align_activation(args, activations[0][name]['output'], activations[1][name]['output'], feature_dim)

        activations_T[name] = {'input': T_var_in, 'output': T_var_out} 
    return activations_T

def fuse_model2(args, model, model1, activations_T, inplace=False, recompute_acts=False, acts_inputs=None, feature_dim=0):
    def rstrip(s, tail):
        if s.endswith(tail):
            return s[0: len(s)-len(tail)]
        return s
    model0 = model if inplace else copy.deepcopy(model)
    aligned_params = {}
    avg_aligned_params = {}
    with torch.no_grad():
        for (layer0_name, layer0_weight), (layer1_name, layer1_weight) in zip(model0.named_parameters(), model1.named_parameters()):
            layer0_weight_shape = layer0_weight.shape
            # name = layer0_name.rstrip(".weight")
            # name = name.rstrip(".bias") # 会误删
            name = rstrip(layer0_name, ".weight")
            name = rstrip(name, ".bias")

            is_conv = len(layer0_weight.shape) == 4
            print(f"\n\n>>> layer name: {name} ||| param name: {layer0_name}")
            print("layer shape:", layer0_weight.shape, layer1_weight.shape)
            if name == "encoder.embed_tokens":
                feature_dim = -1
            else:
                feature_dim = 0

            T_var_in = activations_T[name]['input']
            T_var_out = activations_T[name]['output']

            layer0_aliged = layer0_weight.clone()
            if T_var_in != None and len(layer0_aliged.shape)>1:
                if is_conv:
                    print("zzzz tmp: ", layer0_weight.shape, T_var_in.shape)
                    layer0_aliged = torch.einsum("oiwh,ix->oxwh",layer0_aliged.type(args.dtype), T_var_in)
                else:
                    if feature_dim == 0:
                        layer0_aliged = layer0_aliged.type(args.dtype) @ T_var_in
                    elif feature_dim == -1:
                        layer0_aliged = T_var_in.t() @ layer0_aliged.type(args.dtype) 

            # layer0_aliged = layer0_aliged.view(layer0_weight_shape).contiguous()

            # if recompute_acts and acts_inputs != None:
            #     activations[0] = compute_activations_across_models_zz(args, [model0], acts_inputs, activations)[0]
            #     activation0_update = activations[0][layer0_name.rstrip(".weight")]
            #     print("zzzz-D difference caused by re computations", (activation0-activation0_update).abs().sum())
            
            # layer0_aliged, T_var, _, _ = align_layer2(args, layer0_aliged, layer1_weight, activation0, activation1)
            if feature_dim == 0:
                layer0_aliged = torch.matmul(T_var_out.t(), layer0_aliged.reshape(layer0_weight_shape[0], -1).type(args.dtype)).float()
                # layer0_aliged = (T_var_out.t() @ layer0_aliged.reshape(layer0_weight_shape[0], -1).type(args.dtype)).float()
            elif feature_dim == -1:
                layer0_aliged = torch.matmul(layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype), T_var_out).float()
                # layer0_aliged = (layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype) @ T_var_out).float()

            layer0_aliged = layer0_aliged.view(layer0_weight_shape)

            avg_aligned_params[layer0_name] = (layer0_aliged+layer1_weight)/2
            aligned_params[layer0_name] = layer0_aliged
        utils.update_model(model0, avg_aligned_params)

    aligned_model0 = copy.deepcopy(model0)
    utils.update_model(aligned_model0, aligned_params)
    # geometric_acc, geometric_model = we.get_network_from_param_list(args, avg_aligned_params, test_loader)
    return model0, aligned_model0

def adaptive_fuse_model(args, model, model1, origin_activations, skip_out, skip_in=[], inplace=False, recompute_acts=True, acts_inputs=None, act_feature_dim=-1):
    model0 = model if inplace else copy.deepcopy(model)
    aligned_params = {}
    avg_aligned_params = {}

    activations = copy.deepcopy(origin_activations)
    actual_activations_T = {}

    with torch.no_grad():
        for (layer0_name, layer0_weight), (layer1_name, layer1_weight) in zip(model0.named_parameters(), model1.named_parameters()):
            layer0_weight_shape = layer0_weight.shape
            # name = layer0_name.rstrip(".weight")
            # name = name.rstrip(".bias") # 会误删
            name = rstrip(layer0_name, [".weight", ".bias", ".q_bias", ".v_bias"])

            is_conv = len(layer0_weight.shape) == 4
            print(f"\n\n>>> layer name: {name} ||| param name: {layer0_name}")
            print("layer shape:", layer0_weight.shape)

            if name == "encoder.embed_tokens" or "word_embeddings" in name:
                feature_dim = -1
            else:
                feature_dim = 0

            # T_var_in = activations_T[name]['input']
            # T_var_out = activations_T[name]['output']
            is_conv = (layer0_weight.ndim >= 3)
            if is_conv:
                print("is_conv")
                act_feature_dim = 1
            else:
                act_feature_dim = -1

            T_var_in = None
            T_var_out = None
            if name not in skip_in and layer0_weight.ndim != 1:
                T_var_in, _, _ = align_activation(args, origin_activations[0][name]['input'], activations[0][name]['input'], feature_dim=act_feature_dim)
                # T_var_in = T_var_out
            if name not in skip_out:        
                T_var_out, _, _ = align_activation(args, origin_activations[0][name]['output'], activations[1][name]['output'], feature_dim=act_feature_dim)
            
            if actual_activations_T:
                actual_activations_T[name] = {}
                actual_activations_T[name]['input'] = T_var_in
                actual_activations_T[name]['output'] = T_var_out


            layer0_aliged = layer0_weight.clone()
            if T_var_in != None and len(layer0_aliged.shape)>1:
                if is_conv:
                    print("zzzz tmp: ", layer0_weight.shape, T_var_in.shape)
                    layer0_aliged = torch.einsum("oiwh,ix->oxwh",layer0_aliged.type(args.dtype), T_var_in)
                else:
                    if feature_dim == 0:
                        layer0_aliged = layer0_aliged.type(args.dtype) @ T_var_in
                    elif feature_dim == -1:
                        layer0_aliged = T_var_in.t() @ layer0_aliged.type(args.dtype) 

            # layer0_aliged = layer0_aliged.view(layer0_weight_shape).contiguous()
            # layer0_aliged, T_var, _, _ = align_layer2(args, layer0_aliged, layer1_weight, activation0, activation1)
            
            if T_var_out != None:
                if feature_dim == 0:
                    layer0_aliged = (T_var_out.t() @ layer0_aliged.reshape(layer0_weight_shape[0], -1).type(args.dtype)).float()
                elif feature_dim == -1:
                    # layer0_aliged = torch.matmul(layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype), T_var_out).float()
                    layer0_aliged = (layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype) @ T_var_out).float()

            layer0_aliged = layer0_aliged.view(layer0_weight_shape)

            avg_aligned_params[layer0_name] = (layer0_aliged+layer1_weight)/2
            aligned_params[layer0_name] = layer0_aliged
        
            utils.update_model(model0, aligned_params)

            if recompute_acts and (T_var_in != None or T_var_out != None) and torch.trace(T_var_out)<1-1e-4:
                print('[recompute]')
                activations = compute_activations_across_models_zz(args, (model0, model1), acts_inputs) # , activations=activations

    avged_model = copy.deepcopy(model0)

    utils.update_model(avged_model, avg_aligned_params)

    # geometric_acc, geometric_model = we.get_network_from_param_list(args, avg_aligned_params, test_loader)
    return avged_model, model0, actual_activations_T,activations


def rstrip(s, tails):
    for tail in tails:
        if s.endswith(tail):
            s = s[0: len(s)-len(tail)]
    return s




def adaptive_fuse_model2(args, model, model1, activation_manager, inplace=False, recompute_acts=True, skip_in=[], skip_out=[], p0=0.5):

    model0 = model if inplace else copy.deepcopy(model)
    model1_initial = copy.deepcopy(model1)
    # am = au.ActivationsManager(args, [model0, model1], acts_inputs)
    am: au.ActivationsManager = activation_manager
    am.reset()

    aligned_params = {}
    avg_aligned_params = {}

    # activations = copy.deepcopy(origin_activations)
    actual_activations_T = {rstrip(k, [".weight", ".bias", ".q_bias", ".v_bias"]): {} for k,v in model0.named_parameters()}

    # recomputed_acts = False
    last_module_name = ""
    with torch.no_grad():
        for (layer0_name, layer0_weight), (layer1_name, layer1_weight) in zip(model0.named_parameters(), model1.named_parameters()):
            layer0_weight_shape = layer0_weight.shape
            # name = layer0_name.rstrip(".weight")
            # name = name.rstrip(".bias") # 会误删
            name = rstrip(layer0_name, [".weight", ".bias", ".q_bias", ".v_bias"])

            is_conv = len(layer0_weight.shape) == 4
            print(f"\n\n>>> layer name: {name} ||| param name: {layer0_name}")
            print("layer shape:", layer0_weight.shape, "activations shape")
            if name == "encoder.embed_tokens" or "embeddings" in name:
                feature_dim = -1
            else:
                feature_dim = 0

            # T_var_in = activations_T[name]['input']
            # T_var_out = activations_T[name]['output']
            is_conv = (layer0_weight.ndim >= 3)
            if is_conv:
                print("is_conv")
                act_feature_dim = 1
            else:
                act_feature_dim = -1

            T_var_in = None 
            T_var_out = None 
            if name not in skip_in and layer0_weight.ndim != 1: 
                if "input" in actual_activations_T[name].keys(): 
                    T_var_in = actual_activations_T[name]['input'] 
                else: 
                    T_var_in, _, _ = align_activation(args, am.origin_activations[0][name]['input'], am.activations[0][name]['input'], feature_dim=act_feature_dim) # ??
                    actual_activations_T[name]['input'] = T_var_in 
            if name not in skip_out: 
                if "output" in actual_activations_T[name].keys(): 
                    T_var_out = actual_activations_T[name]['output'] 
                else: 
                    T_var_out, _, _ = align_activation(args, am.activations[0][name]['output'], am.activations[1][name]['output'], feature_dim=act_feature_dim) 
                    # T_var_out, _, _ = align_activation(args, am.origin_activations[0][name]['output'], am.activations[1][name]['output'], feature_dim=act_feature_dim)
                    # T_var_out, _, _ = align_activation(args, am.activations[0][name]['output'], am.activations[1][name]['output'], feature_dim=act_feature_dim)
                    actual_activations_T[name]['output'] = T_var_out 

            layer0_aliged = layer0_weight.clone()
            if T_var_in != None and len(layer0_aliged.shape) > 1:
                print("[align in]")
                if is_conv:
                    print("zzzz conv: ", layer0_weight.shape, T_var_in.shape)
                    layer0_aliged = torch.einsum("oiwh,ix->oxwh",layer0_aliged.type(args.dtype), T_var_in)
                else:
                    if feature_dim == 0:
                        layer0_aliged = layer0_aliged.type(args.dtype) @ T_var_in
                    elif feature_dim == -1:
                        layer0_aliged = T_var_in.t() @ layer0_aliged.type(args.dtype) 

            # layer0_aliged = layer0_aliged.view(layer0_weight_shape).contiguous()
            # layer0_aliged, T_var, _, _ = align_layer2(args, layer0_aliged, layer1_weight, activation0, activation1)
            
            if T_var_out != None:
                print("[align out]")
                if feature_dim == 0:
                    layer0_aliged = (T_var_out.t() @ layer0_aliged.reshape(layer0_weight_shape[0], -1).type(args.dtype)).float()
                elif feature_dim == -1:
                    # layer0_aliged = torch.matmul(layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype), T_var_out).float()
                    layer0_aliged = (layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype) @ T_var_out).float()
                else:
                    raise NotImplementedError("feature_dim only suppor 0 or -1 now")
            layer0_aliged = layer0_aliged.view(layer0_weight_shape)
            if not layer0_name.split('.')[1] in ['embed_tokens', 'embed_positions']:
                # print('>> {}, {}, {}, {}, {}.'.format(avg_aligned_params, layer0_name, layer0_aliged, p0, layer1_weight))
                avg_aligned_params[layer0_name] = layer0_aliged * p0 + layer1_weight * ( 1 - p0 )
            aligned_params[layer0_name] = layer0_aliged
        
            if recompute_acts and p_dis(layer0_aliged, layer0_weight)>1e-6: #(torch.trace(T_var_out)<(1-1e-4)*T_var_out.size(0) or torch.trace(T_var_in)<(1-1e-4)*T_var_in.size(0)):
                utils.update_model(model0, aligned_params)      # layer0_wight will be modified
                am.update_model(0, layer0_name, layer0_aliged)
                # recomputed_acts = True
                print('[recompute]')
                am.recompute(last_module_name, name)
                last_module_name = name
                in_dis = -1
                # if 'input' in am.activations[0][name].keys():
                #     _, in_dis = p_dis(am.origin_activations[0][name]['input'], am.activations[0][name]['intput'], args)
                _, out_dis = p_dis(am.activations[0][name]['output'], am.activations[1][name]['output'], args)
                print("distance after recompute: ",  out_dis)
                # am.activations = compute_activations_across_models_zz(args, (model0, model1), am.inputs) # , activations=activations
            else:
                recomputed_acts = False
    avged_model = copy.deepcopy(model1_initial)
    utils.update_model(avged_model, avg_aligned_params)

    # geometric_acc, geometric_model = we.get_network_from_param_list(args, avg_aligned_params, test_loader)
    return avged_model, model0, actual_activations_T, am

def activations_sorts(activation, feature_dim=0):
    axis = list(range(activation.ndim))
    if feature_dim < 0:
        feature_dim = activation.ndim + feature_dim
    axis.remove(feature_dim)
    # activation = activation.sum(axis=axis)
    activation = activation.std(axis=axis)

    values, indices = torch.sort(activation)
    return indices

def unify_align_model(args, model, inplace=False, acts_inputs=None, act_feature_dim=-1, skip_in=[], skip_out=[]):
    def rstrip(s, tail):
        if s.endswith(tail):
            return s[0: len(s)-len(tail)]
        return s
    model0 = model if inplace else copy.deepcopy(model)
    aligned_params = {}
    am = au.ActivationsManager(args, [model0], acts_inputs)
    
    args.mem_efficient = True
    last_module_name = ""
    Ts = {}
    
    indices = None

    with torch.no_grad():
        for (layer0_name, layer0_weight) in model0.named_parameters():
            layer0_weight_shape = layer0_weight.shape
            name = rstrip(layer0_name, ".weight")
            name = rstrip(name, ".bias")

            if name == "encoder.embed_tokens" or "embedding" in name:
                feature_dim = -1
            else:
                feature_dim = 0
            print(f"\n\n>>> layer name: {name} ||| param name: {layer0_name}")
            print("layer shape:", layer0_weight.shape, "feature_dim_size:", layer0_weight.shape[feature_dim])
            
            # T_var_in = activations_T[name]['input']
            # T_var_out = activations_T[name]['output']
            is_conv = (layer0_weight.ndim == 4)
            if is_conv:
                print("is_conv")
                act_feature_dim = 1
            else:
                act_feature_dim = -1

            if name in skip_in:
                T_var_in = None
            else:
                am.origin_activations[0][name]['input']
                am.activations[0][name]['input']
                T_var_in, _, _ = align_activation(args, am.origin_activations[0][name]['input'], am.activations[0][name]['input'], feature_dim=act_feature_dim)

            layer0_aliged = layer0_weight.clone()
            
            if T_var_in != None and len(layer0_aliged.shape)>1:
                print("align input, T_var_in.shape: ", T_var_in.shape)

                if is_conv:
                    layer0_aliged = torch.einsum("oiwh,ix->oxwh",layer0_aliged.type(args.dtype), T_var_in)
                else:
                    if feature_dim == 0:
                        layer0_aliged = layer0_aliged.type(args.dtype) @ T_var_in
                    elif feature_dim == -1:
                        layer0_aliged = T_var_in.t() @ layer0_aliged.type(args.dtype) 

            if name in skip_out:
                print(f"Warning: {layer0_name}'s feature_dim size=", layer0_aliged.shape[feature_dim])
            else:
                indices = activations_sorts(am.origin_activations[0][name]['output'], feature_dim=act_feature_dim)
                if feature_dim == -1:
                    layer0_aliged = layer0_aliged.transpose(0, -1)
                                
                layer0_aliged = layer0_aliged[indices]
                print("align output, output permute indices shape:", indices.shape)
                
                if feature_dim == -1:
                    layer0_aliged = layer0_aliged.transpose(0, -1)
                layer0_aliged = layer0_aliged.view(layer0_weight_shape)

            Ts[layer0_name] = {'input': T_var_in, 'output': indices}
            aligned_params[layer0_name] = layer0_aliged
            
            utils.update_model(model0, aligned_params)

            am.activations = compute_activations_across_models_zz(args, [model0 ], acts_inputs) # , activations=activations

    return model0, Ts


def args_modify(args, act_num_samples=1000,
                geom_ensemble_type='acts',
                importance="uniform",
                importance_method="softmax",
                softmax_temperature="10",
                dtype=torch.float64,
                mem_efficient = False,
                exact=True,
                reg = 0.1
                ):
    args.act_num_samples = act_num_samples
    args.geom_ensemble_type = geom_ensemble_type

    args.importance = importance
    args.importance_method = importance_method
    args.softmax_temperature = softmax_temperature
    args.dtype = dtype
    args.mem_efficient = mem_efficient

    args.exact = exact
    args.reg = reg


def process_activations_T(activations_T, mode="None"):
    """
    args:
        mode: in | out | None
    """
    processed_a_T = copy.deepcopy(activations_T)
    before_k = None
    for k, v in processed_a_T.items():
        if before_k != None:
            if mode == "in":
                v['input'] = processed_a_T[before_k]['output']
            elif mode == "out":
                processed_a_T[before_k]['output'] = v['input']
        before_k = k
    return processed_a_T


def save_fused_model(fused_model, path, model0_path=None):
    checkpoint = {}
    if model0_path != None:
        checkpoint = torch.load(model0_path)
    
    checkpoint['model'] = fused_model.state_dict()
    torch.save(checkpoint, path)


def tune_activations_T_for_transformers(activations_T_x):
    activations_T = copy.deepcopy(activations_T_x)
    for layer_idx in range(6):
        if layer_idx==0:
            prev_last_module = 'embed_tokens'
        else:
            prev_last_module = f'layers.{layer_idx-1}.final_layer_norm'

        activations_T[f'encoder.layers.{layer_idx}.self_attn.q_proj']['input'] = activations_T[f'encoder.{prev_last_module}']['output']
        activations_T[f'encoder.layers.{layer_idx}.self_attn.k_proj']['input'] = activations_T[f'encoder.{prev_last_module}']['output']
        activations_T[f'encoder.layers.{layer_idx}.self_attn.v_proj']['input'] = activations_T[f'encoder.{prev_last_module}']['output']

        activations_T[f'decoder.layers.{layer_idx}.self_attn.q_proj']['input'] = activations_T[f'decoder.{prev_last_module}']['output']
        activations_T[f'decoder.layers.{layer_idx}.self_attn.k_proj']['input'] = activations_T[f'decoder.{prev_last_module}']['output']
        activations_T[f'decoder.layers.{layer_idx}.self_attn.v_proj']['input'] = activations_T[f'decoder.{prev_last_module}']['output']

        activations_T[f'encoder.layers.{layer_idx}.self_attn.q_proj']['output'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.k_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.self_attn.q_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.k_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn.q_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn.k_proj']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.self_attn.v_proj']['output'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.k_proj']['output']  # 感觉不需要
        activations_T[f'decoder.layers.{layer_idx}.self_attn.v_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.k_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn.v_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn.k_proj']['output'] 
        
        activations_T[f'encoder.layers.{layer_idx}.self_attn.out_proj']['input'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.v_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.self_attn.out_proj']['input'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.v_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn.out_proj']['input'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn.v_proj']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.self_attn_layer_norm']['input'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.out_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.self_attn_layer_norm']['input'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.out_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn_layer_norm']['input'] - activations_T[f'decoder.layers.{layer_idx}.encoder_attn.out_proj']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.fc1']['input'] = activations_T[f'encoder.layers.{layer_idx}.self_attn_layer_norm']['output']  # 本来就差不多
        activations_T[f'decoder.layers.{layer_idx}.fc1']['input'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn_layer_norm']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.fc2']['input'] = activations_T[f'encoder.layers.{layer_idx}.fc1']['output']
        activations_T[f'decoder.layers.{layer_idx}.fc2']['input'] = activations_T[f'decoder.layers.{layer_idx}.fc1']['output']

        activations_T[f'encoder.layers.{layer_idx}.final_layer_norm']['input'] = activations_T[f'decoder.layers.{layer_idx}.fc2']['output']
        activations_T[f'decoder.layers.{layer_idx}.final_layer_norm']['input'] = activations_T[f'decoder.layers.{layer_idx}.fc2']['output']

    activations_T['decoder.output_projection']['input'] = activations_T[f'decoder.layers.5.final_layer_norm']['output']
    return activations_T


def check_activations_T(activations_T):

    print("non diag sum:")
    for key in activations_T.keys():
        try:
            input_diag = activations_T[key]['input'].diag().sum()
            output_diag = activations_T[key]['output'].diag().sum()
            input_non_diag = activations_T[key]['input'].size(0) - input_diag
            output_non_diag = activations_T[key]['output'].size(0) - output_diag
            print(f"{key}: {input_non_diag}, {output_non_diag}")
        except Exception as e:
            print(key, e)

def check_activations(activations):
    """
    compare the difference of two models' activations
    """
    print("input_diff, output_diff, input_sum_diff, output_sum_diff")
    for key in activations[0].keys():
        try:
            print(f"{key}: input shape, {activations[0][key]['input'][0].shape}, output shape{activations[0][key]['output'][0].shape}")

        except Exception as e:
            print("error", key, e)

def check_model_parameters(model, activation_manager):
    for name, p in model.named_parameters():
        m_name = rstrip(name, ['.weight', '.bias'])
        print(f"\n>>> {m_name}")
        try:
            print(activation_manager.activations[0][m_name]['input'].shape, activation_manager.activations[0][m_name]['output'].shape)
        except Exception as e:
            print(e)


def dis(t1, t2, tag="", method="diff"):

    def _pairwise_distances(x, y):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist = torch.clamp(dist, min=0.0)
        dist = dist ** (1/2)
        return dist
    if method == 'diff':
        print(tag, (t1-t2).abs().sum()/(t1.abs().sum()+t2.abs().sum()))
    else:
        return _pairwise_distances(t1, t2)


def model_param_diff(net1, net2, thred=0.02):
    for (name, p1), (_, p2) in zip(net1.named_parameters(), net2.named_parameters()):
        d = (p1-p2).abs().sum()/(p1.abs()+p2.abs()).sum()
        if d > thred:
            print(name, d, p1.shape)


def activation_diff(activationManager):
    for name in activationManager.origin_activations[0].keys():
        d1, d2 = -1, -1
        if type(activationManager.origin_activations[0][name]['input']) == torch.Tensor:
            op1 = activationManager.origin_activations[0][name]['input']
            op2 = activationManager.origin_activations[1][name]['input']
            d1 = (op1-op2).abs().sum()/(op1.abs().sum()+op2.abs().sum())
        if 'output' in activationManager.origin_activations[0][name].keys() and type(activationManager.origin_activations[0][name]['output']) == torch.Tensor:
            op1 = activationManager.origin_activations[0][name]['output']
            op2 = activationManager.origin_activations[1][name]['output']
            d2 = (op1-op2).abs().sum()/(op1.abs().sum()+op2.abs().sum())
        print(name, d1, d2)


def p_dis(op1, op2, args=None, feature_dim=-1):
    if args!=None:
        activation0, activation1 = op1, op2
        if feature_dim == 1:
            # 把当前层feature维放到最前面(原文是先把batch维放到了最后), 
            # 这个做法更慢，并且两种算法目前都后续过程都没有影响 
            activation0 = torch.einsum('bf...->fb...', activation0)
            activation1 = torch.einsum('bf...->fb...', activation1)
            ## 原做法
            # activation0 = activation0.permute([*range(1, len(activation0.shape)), 0]).contiguous()
            # activation1 = activation1.permute([*range(1, len(activation1.shape)), 0]).contiguous()
        elif feature_dim == -1:
            activation0 = activation0.reshape(-1, activation0.shape[-1]).T
            activation1 = activation1.reshape(-1, activation1.shape[-1]).T
        activation0 = activation0.reshape(activation0.shape[0], -1).contiguous()
        activation1 = activation1.reshape(activation1.shape[0], -1).contiguous()
        # print("activation0.shape:", activation0.shape, "activation1.shape:", activation1.shape)    
        M0 = gm.GroundMetric.PROCESS(args, activation0, activation1)    
        d = torch.trace(M0)
        return M0, d
    else:
        d = (op1-op2).abs().sum()/(op1.abs().sum()+op2.abs().sum())
    return d


if __name__ == '__main__':
    import prepare 
    # args, model0, model1, activations, train_loader, test_loader = cifar_resnet_test()
    args, model0, model1, activations, train_loader, test_loader, acts_inputs = prepare.mnist_test(version=2)
    fused_model, aligned_model, actural_acts_T, new_activation = adaptive_fuse_model(args, model0, model1, activations, acts_inputs=acts_inputs, recompute_acts=True, skip_out=['fc4'])
    _ = utils.test(args, aligned_model, test_loader)
    # activations_T = align_activations(args, activations)
    # fused_model, aligned_params = fuse_model2(args, model0, model1, activations_T)
    # utils.test(args, fused_model, test_loader)
    # print(activations)
    # args, hf_models, activations, act_inputs = uz.bert_test()
    # unify_aligned_model = unify_align_model(args, hf_models[0], activations, acts_inputs=act_inputs)
    # args, models, configs, tokenizers, activations, acts_inputs = prepare.deberta_test(tag='deberta-base')
    # fused_model, aligned_model, actural_acts_T, am = adaptive_fuse_model2(args, models[0], models[1], acts_inputs, 
    #     recompute_acts=True, skip_in=['deberta.embeddings.word_embeddings'], skip_out=['cls.predictions.decoder', 'cls.predictions'])
    # prepare.test_mlm(fused_model, tokenizers[0])
    # prepare.test_mlm(aligned_model, tokenizers[0])
    pass
