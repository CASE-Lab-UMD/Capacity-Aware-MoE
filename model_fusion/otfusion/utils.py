import os
import pickle
from itertools import chain
import csv
import collections, torch
import numpy as np
import otfusion.routines as routines
from otfusion.data import get_dataloader
import otfusion.cifar.train as cifar_train
import otfusion.cifar.models.vgg as vgg
import otfusion.compute_activations as compute_activations


def get_timestamp_other():
    import time
    import datetime
    ts = time.time()
    # %f allows granularity at the micro second level!
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S_%f')
    return timestamp


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def pickle_obj(obj, path, mode = "wb", protocol=pickle.HIGHEST_PROTOCOL):
    '''
    Pickle object 'obj' and dump at 'path' using specified
    'mode' and 'protocol'
    Returns time taken to pickle
    '''

    import time
    st_time = time.perf_counter()
    pkl_file = open(path, mode)
    pickle.dump(obj, pkl_file, protocol=protocol)
    end_time = time.perf_counter()

    return (end_time - st_time)


def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))


def save_results_params_csv(path, results_dic, args, ordered=True):
    if os.path.exists(path):
        add_header = False
    else:
        add_header = True

    with open(path, mode='a') as csv_file:

        if args.deprecated is not None:
            params = args
        else:
            params = vars(args)

        # Merge with params dic
        if ordered:
            # sort the parameters by name before saving
            params = collections.OrderedDict(sorted(params.items()))

        results_and_params_dic = dict_union(results_dic, params)

        writer = csv.DictWriter(csv_file, fieldnames=results_and_params_dic.keys())

        # Add key header if file doesn't exist
        if add_header:
            writer.writeheader()

        # Add results and params record
        writer.writerow(results_and_params_dic)


def isnan(x):
    return x != x


def get_model_activations(args, models, unit_batch_train_loader=None):

    if args.activation_histograms and args.act_num_samples > 0:
        activations = compute_activations.compute_activations_across_models(models,
                                                unit_batch_train_loader, args.act_num_samples)
    else:
        activations = None

    return activations


def get_model_layers_cfg(model_name):
    print('model_name is ', model_name)
    if model_name == 'mlpnet' or model_name[-7:] =='encoder':
        return None
    elif model_name[0:3].lower()=='vgg':
        cfg_key = model_name[0:5].upper()
    elif model_name[0:6].lower() == 'resnet':
        return None
    return vgg.cfg[cfg_key]


def _get_config(args):
    print('refactored get_config')
    import hyperparameters.vgg11_cifar10_baseline as cifar10_vgg_hyperparams  # previously vgg_hyperparams
    import hyperparameters.vgg11_half_cifar10_baseline as cifar10_vgg_hyperparams_half
    import hyperparameters.vgg11_doub_cifar10_baseline as cifar10_vgg_hyperparams_doub
    import hyperparameters.vgg11_quad_cifar10_baseline as cifar10_vgg_hyperparams_quad
    import hyperparameters.resnet18_nobias_cifar10_baseline as cifar10_resnet18_nobias_hyperparams
    import hyperparameters.resnet18_nobias_nobn_cifar10_baseline as cifar10_resnet18_nobias_nobn_hyperparams
    import hyperparameters.mlpnet_cifar10_baseline as mlpnet_hyperparams

    config = None
    second_config = None

    if args.dataset.lower() == 'cifar10':
        if args.model_name == 'mlpnet':
            config = mlpnet_hyperparams.config
        elif args.model_name == 'vgg11_nobias':
            config = cifar10_vgg_hyperparams.config
        elif args.model_name == 'vgg11_half_nobias':
            config = cifar10_vgg_hyperparams_half.config
        elif args.model_name == 'vgg11_doub_nobias':
            config = cifar10_vgg_hyperparams_doub.config
        elif args.model_name == 'vgg11_quad_nobias':
            config = cifar10_vgg_hyperparams_quad.config
        elif args.model_name == 'resnet18_nobias':
            config = cifar10_resnet18_nobias_hyperparams.config
        elif args.model_name == 'resnet18_nobias_nobn':
            config = cifar10_resnet18_nobias_nobn_hyperparams.config
        else:
            raise NotImplementedError
    elif args.dataset.lower() == 'mnist':
        config = {'dataset': 'mnist', 'model': args.model_name, 'optimizer': 'SGD', 
                    'optimizer_decay_at_epochs': [30, 60, 90, 120, 150, 180, 210, 240, 270], 'optimizer_decay_with_factor': 2.0, 
                    'optimizer_learning_rate': 0.05, 'optimizer_momentum': 0.9, 'optimizer_weight_decay': 0.0005, 'batch_size': 128, 'num_epochs': 300, 'seed': 42}

    if hasattr(args, 'second_model_name') and args.second_model_name is not None:
        if 'vgg' in args.second_model_name:
            if 'half' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_half.config
            elif 'doub' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_doub.config
            elif 'quad' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_quad.config
            elif args.second_model_name == 'vgg11_nobias':
                second_config = cifar10_vgg_hyperparams.config
            else:
                raise NotImplementedError
        elif 'resnet' in args.second_model_name:
            if args.second_model_name == 'resnet18_nobias':
                second_config= cifar10_resnet18_nobias_hyperparams.config
            elif args.second_model_name == 'resnet18_nobias_nobn':
                config = cifar10_resnet18_nobias_nobn_hyperparams.config
            else:
                raise  NotImplementedError
    else:
        second_config = config

    return config, second_config

def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fuse_with_various_weights_test(args, model, params0, params1, test_loader):
    params1 = dict(params1)
    accs = {}
    for avg_weight in np.linspace(0,1, 11):
        avg_aligned_layers = {}
        for k,v in params0.items():
            avg_aligned_layers[k] = (1 - avg_weight) * params0[k] + \
                           avg_weight * params1[k]
        update_model(model, avg_aligned_layers)
        acc = routines.test(args, model, test_loader, {'test_losses':[]})
        print(f"---------- avg with weight={avg_weight} ----------")
        print(f"acc={acc}")
        accs[avg_weight] = acc
    return accs

def save_activations(model_name, activation, dump_path):
    mkdir(dump_path)
    pickle_obj(
        activation,
        os.path.join(dump_path, 'model_{}_activations'.format(model_name))
    )

def get_importance_hist(importance, method, softmax_temperature=1, eps=1e-9, dtype=torch.float64):
    # assert len(importance.shape) == 2
    importance = importance.reshape(importance.shape[0], -1)
    importance = importance.abs().type(dtype)
    if method == "softmax":
        # importance = torch.einsum("f...-> f", importance.abs())
        importance = importance.abs().mean(axis=list(range(1, importance.ndim)))
        importance_hist = torch.softmax(importance / softmax_temperature, dim=0)
    elif method == 'l1':
        importance_hist = torch.linalg.norm(importance, ord=1, axis=-1)
    elif method == 'l2':
        importance_hist = torch.linalg.norm(importance, ord=2, axis=-1)
    else:
        raise NotImplementedError(f"importance method {method} not found") 
    importance_hist = importance_hist / importance_hist.sum()
    # assert abs(importance_hist.sum()-1) < 1e-7
    return importance_hist 
    


def get_histogram(args, cardinality, layer_weight = None, 
                    activation=None, dtype=torch.float64):
    if args.importance == 'uniform':
        # print("uniform importance")
        hist =  torch.ones(cardinality, dtype=dtype)/cardinality
    elif args.importance == 'wts':
        assert layer_weight != None, "layer weight can't be none"
        print(f"get importance from weights with method {args.importance_method}")
        hist =  get_importance_hist(layer_weight.reshape(layer_weight.shape[0], -1), args.importance_method, dtype=dtype)
    elif args.importance == 'acts':
        assert activation != None, "activation can't be none"
        print(f"get importance from activations with method {args.importance_method}")
        hist = get_importance_hist(activation, args.importance_method, args.softmax_temperature, dtype=dtype)
    else:
        raise NotImplementedError(f"importance {args.importance} not found") 
    assert hist.max()<1-1e-3, "distribution of importance error"
    return hist.type(args.dtype).detach().cpu()


def update_model(model, update_params):
    state = model.state_dict()
    for k, v in update_params.items():
        state[k] = v
    model.load_state_dict(state)


def test(args, network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    wrong_results = []
    categories_correct = {}
    categories_num = {}
    categories_pred_num = {}

    #   with torch.no_grad():
    for data, target in test_loader:
        # print(data.shape, target.shape)
        # if len(target.shape)==1:
        #     data = data.unsqueeze(0)
        #     target = target.unsqueeze(0)
        # print(data, target)
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)

        output = network(data)
        # if debug:
        #     print("output is ", output)

        # if args.dataset.lower() == 'cifar10':
        #     # mnist models return log_softmax outputs, while cifar ones return raw values!
        #     test_loss += cifar_criterion(output, target).item()
        # elif args.dataset.lower() == 'mnist':
        #     test_loss += F.nll_loss(output, target, size_average=False).item()

        cat = target.item()
        pred = output.data.max(1, keepdim=False)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        categories_num[cat] = categories_num.get(cat, 0) +1
        categories_pred_num[pred.item()] = categories_pred_num.get(pred.item(), 0) +1
        categories_correct[cat] = categories_correct.get(cat, 0) + (pred.item() == cat)

        if not pred.item() == cat:
            wrong_results.append((data, target))

    print("size of test_loader dataset: ", len(test_loader.dataset))
    # test_loss /= len(test_loader.dataset)

    ans = (float(correct) * 100.0) / len(test_loader.dataset)

    print("total accaracy:", ans)
    for cat in sorted(categories_num.keys()):
        print(f"category {cat}: prec({categories_correct[cat]}/{categories_num[cat]})={categories_correct[cat]/categories_num[cat]:.3f}\trecall({categories_correct[cat]}/{categories_pred_num.get(cat, 0)})={categories_correct[cat]/categories_pred_num.get(cat, 0.1):.3f}")
    
    # return wrong_results
    # if not return_loss:
    #     return ans
    # else:
    #     return ans, test_loss


def retrain_models(args, old_networks, train_loader, test_loader, config, tensorboard_obj=None, initial_acc=None, nicks=None):
    accuracies = []
    retrained_networks = []
    for i in range(len(old_networks)):
        nick = nicks[i]
        print("Retraining model : ", nick)

        if initial_acc is not None:
            start_acc = initial_acc[i]
        else:
            start_acc = -1

        if args.dataset.lower()[0:7] == 'cifar10':    
            retrain_loader = train_loader

            output_root_dir = f"{ROOT}/checkpoints/saved_models/{args.rt_exp_name}"
            output_root_dir = os.path.join(output_root_dir, nick)
            os.makedirs(output_root_dir, exist_ok=True)

            retrained_network, acc = cifar_train.get_retrained_model(args, retrain_loader, test_loader, old_networks[i], config, output_root_dir, tensorboard_obj=tensorboard_obj, nick=nick, start_acc=initial_acc[i])
        elif args.dataset.lower() == 'mnist':

            retrain_loader = train_loader 
            retrained_network, acc = get_retrained_model(args, retrain_loader, test_loader, old_network=old_networks[i], tensorboard_obj=tensorboard_obj, nick=nick, start_acc=start_acc, retrain_seed=args.rt_seed)
        retrained_networks.append(retrained_network)
        accuracies.append(acc)
    return retrained_networks, accuracies

def get_retrained_model(args, train_loader, test_loader, old_network, tensorboard_obj=None, nick='', start_acc=-1, retrain_seed=-1):
    from torch import optim
    torch.backends.cudnn.enabled = False
    # if args.rt_lr_decay > 0:
    #     args.rt_lr = args.lr / args.rt_lr_decay
    
    print('optimizer_learning_rate is ', args.rt_lr)
    if retrain_seed!=-1:
        torch.manual_seed(retrain_seed)
        
    optimizer = optim.SGD(old_network.parameters(), lr=args.rt_lr, momentum=args.rt_momentum)

    log_dict = {}
    log_dict['train_losses'] = []
    log_dict['train_counter'] = []
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]

    acc = routines.test(args, old_network, test_loader)
    print("check accuracy once again before retraining starts: ", acc)
    if start_acc == -1:
        start_acc = acc
    if tensorboard_obj is not None and start_acc != -1:
        tensorboard_obj.add_scalars(f'test_accuracy_percent/', {nick: start_acc}, global_step=0)


    best_acc = -1
    for epoch in range(1, args.rt_epoch + 1):
        routines.train(args, old_network, optimizer, train_loader, log_dict, epoch, model_id=epoch)
        acc, loss = routines.test(args, old_network, test_loader,  return_loss=True)

        if tensorboard_obj is not None:
            assert nick != ''
            tensorboard_obj.add_scalars(f'test_loss/', {nick: loss}, global_step=epoch)
            tensorboard_obj.add_scalars(f'test_accuracy_percent/', {nick: acc}, global_step=epoch)

        print("At retrain epoch the accuracy is : ", acc)
        best_acc = max(best_acc, acc)

    return old_network, best_acc

def load_ft_models(task="mnli", tag='bert-base-uncased'):
    path = f"/public/data0/NLP/zhangzheng15/transformers/checkpoints/{tag}/pytorch_model/{task}/pytorch_model.bin"
    model_checkpoint = torch.load(path, map_location=torch.device('cpu'))
    names = model_checkpoint.keys()
    non_layer_names = [name for name in names if not name.startswith('bert.encoder.layer')]
    print(non_layer_names)
    non_bert_names = [name for name in names if not name.startswith('bert')]
    for name in non_bert_names:
        print(name, ":", model_checkpoint[name].shape)
    return model_checkpoint


def rstrip(s, tails):
    for tail in tails:
        if s.endswith(tail):
            s = s[0: len(s)-len(tail)]
    return s

import otfusion.activation_utils as au
import otfusion.ground_metric as gm
import copy


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


def adaptive_fuse_model(args, model, model1, activation_manager,
                         inplace=False, recompute_acts=True, skip_in=[], skip_out=[], p0=0.5):
    model0 = model if inplace else copy.deepcopy(model)
    model1_initial = copy.deepcopy(model1)
    # am = au.ActivationsManager(args, [model0, model1], acts_inputs)
    am: au.ActivationsManager = activation_manager
    am.reset()

    aligned_params = {}
    avg_aligned_params = {}

    # activations = copy.deepcopy(origin_activations)
    actual_activations_T = {rstrip(k, [".weight", ".bias", ".q_bias", ".v_bias"]): {} for k, v in
                            model0.named_parameters()}

    act_feature_dim = -1
    last_module_name = ""
    with torch.no_grad():
        for (layer0_name, layer0_weight), (layer1_name, layer1_weight) in zip(model0.named_parameters(),
                                                                              model1.named_parameters()):
            layer0_weight_shape = layer0_weight.shape
            name = rstrip(layer0_name, [".weight", ".bias", ".q_bias", ".v_bias", ".word_embeddings"])


            print(f"\n\n>>> layer name: {name} ||| param name: {layer0_name}")
            print("layer shape:", layer0_weight.shape, "activations shape")
            if name == "encoder.embed_tokens" or "embeddings" in name:
                feature_dim = -1
            else:
                feature_dim = 0

            T_var_in = None
            T_var_out = None
            if name not in skip_in and layer0_weight.ndim != 1:
                if "input" in actual_activations_T[name].keys():
                    T_var_in = actual_activations_T[name]['input']
                else:
                    T_var_in, _, _ = align_activation(args, am.origin_activations[0][name]['input'],
                                                      am.activations[0][name]['input'],
                                                      feature_dim=act_feature_dim)  # ??
                    actual_activations_T[name]['input'] = T_var_in
            if name not in skip_out:
                if "output" in actual_activations_T[name].keys():
                    T_var_out = actual_activations_T[name]['output']
                else:
                    T_var_out, _, _ = align_activation(args, am.activations[0][name]['output'],
                                                       am.activations[1][name]['output'], feature_dim=act_feature_dim)
                    actual_activations_T[name]['output'] = T_var_out

            layer0_aliged = layer0_weight.clone()
            if T_var_in != None and len(layer0_aliged.shape) > 1:
                print("[align in]")
                if feature_dim == 0:
                    layer0_aliged = layer0_aliged.type(args.dtype) @ T_var_in
                elif feature_dim == -1:
                    layer0_aliged = T_var_in.t() @ layer0_aliged.type(args.dtype)

            if T_var_out != None:
                print("[align out]")
                if feature_dim == 0:
                    layer0_aliged = (T_var_out.t() @ layer0_aliged.reshape(layer0_weight_shape[0], -1).type(
                        args.dtype)).float()
                elif feature_dim == -1:
                    layer0_aliged = (
                                layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype) @ T_var_out).float()
                else:
                    raise NotImplementedError("feature_dim only suppor 0 or -1 now")
            layer0_aliged = layer0_aliged.view(layer0_weight_shape)
            if not layer0_name.split('.')[1] in ['embed_tokens', 'embed_positions']:
                avg_aligned_params[layer0_name] = layer0_aliged * p0 + layer1_weight * (1 - p0)
            aligned_params[layer0_name] = layer0_aliged

            if recompute_acts and p_dis(layer0_aliged,
                                        layer0_weight) > 1e-6:
                utils.update_model(model0, aligned_params)
                am.update_model(0, layer0_name, layer0_aliged)
                print('[recompute]')
                am.recompute(last_module_name, name)
                last_module_name = name
                _, out_dis = p_dis(am.activations[0][name]['output'], am.activations[1][name]['output'], args)
                print("distance after recompute: ", out_dis)
            else:
                recomputed_acts = False
    avged_model = copy.deepcopy(model1_initial)
    utils.update_model(avged_model, avg_aligned_params)

    return avged_model, model0, actual_activations_T, am


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
    print(
        f"Ratio of trace to the matrix sum:  {(torch.trace(T) / torch.sum(T)).item():.4f};\t otcost: , {ot_cost.item():.2f};\t distance:{dis}")

    # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))

    return T, mu, nu


if __name__ == "__main__":
    from utils_zz import mnist_test
    from utils_zz import cifar_vgg_test

    import utils_zz as uz
    from tensorboardX import SummaryWriter
    import utils
    args, model0, model1, activations, train_loader, test_loader, acts_inputs = cifar_vgg_test()
    config, second_config = utils._get_config(args)
    args.rt_exp_name = f"{args.dataset}-{args.model_name}"

    tensorboard_obj = SummaryWriter(log_dir=f"{ROOT}/checkpoints/runs/{args.rt_exp_name}")

    fused_model, aligned_model = uz.fuse_model(args, model0, model1, activations, inplace=False)
    activation_gradients1, fisher_infos = uz.compute_activations_gredients_across_models(args, [model0, model1], acts_inputs)
    fisher_fused_model = uz.avg_model(aligned_model, model1, fisher_infos[0], fisher_infos[1])

    print(routines.test(args, fused_model, test_loader))
    print(routines.test(args, fisher_fused_model, test_loader))
    
    initial_acc = [81.73, 81.04, 90.3,90.5]
    train_loader, test_loader = get_dataloader(args, args.rt_batch_size, args.rt_batch_size)
    retrain_models(args, [fused_model, fisher_fused_model, model0, model1], train_loader, test_loader, config, tensorboard_obj=tensorboard_obj, initial_acc=initial_acc, nicks=['model0', "fused_model", "fisher_fused_model", "model1"])