import ot
import torch
import numpy as np
import otfusion.routines as routines
from otfusion.model import get_model_from_name
import otfusion.utils as utils
from otfusion.ground_metric import GroundMetric
import math


def get_importance_hist_from_weights(layer_weight, method="l2", eps=1e-9):
    layer = layer_weight.detach().cpu()
    layer = layer.contiguous().view(layer_weight.shape[0], -1).numpy()

    if method == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
                    np.float64) + eps
    elif method == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
                    np.float64) + eps
    else:
        raise NotImplementedError
    importance_hist = (importance_hist/importance_hist.sum())

    assert (importance_hist.sum() - 1.0)<1e-9
    return importance_hist


def get_importance_hist_from_activations(activation, softmax_temperature=10, dtype=np.float64):
    # return softmax over the activations raised to a temperature
    # layer_name is like 'fc1.weight', while activations only contains 'fc1'
    activation = torch.einsum("f...-> f", activation)
    # activation = activation.sum(axis=list(range(1,activation.ndim)))
    importance_hist =  torch.softmax(activation / softmax_temperature, dim=0).data.cpu().numpy().astype(dtype)
    assert (importance_hist.sum() - 1.0) < 1e-9
    return importance_hist


def get_histogram(args, cardinality, layer_weight = None, is_conv=False, method="l2", 
                    activation=None, dtype=np.float64):
    if activation is None and layer_weight is None:
        return np.ones(cardinality)/cardinality
    else:
        assert not (activation != None and layer_weight != None), "can only calculate importance hist from weight or activation"
        if layer_weight != None:
            return get_importance_hist_from_weights(layer_weight, is_conv, method)
        if activation != None:
            return get_importance_hist_from_activations(activation, args.softmax_temperature)


def get_wassersteinized_layers_modularized(args, networks, activations=None, skip_layers=[], eps=1e-7):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    avg_aligned_layers = {}
    T_var = None
    ground_metric_object = GroundMetric(args)
    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    index = 0
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

        # todo skip the embedding layers
        skip = False
        for layer in skip_layers:
            if layer in layer0_name:
                skip = True
                break
        if skip:
            continue

        assert fc_layer0_weight.shape == fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]
        fc_layer0_weight_data = fc_layer0_weight.data
        fc_layer1_weight_data = fc_layer1_weight.data

        # todo detect the 1st layer
        if index == 0:
            M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
            aligned_wt = fc_layer0_weight_data
            index += 1
        else:
            # aligned_wt = None, this caches the tensor and causes OOM
            if fc_layer1_weight.data.shape[1] != T_var.shape[0]:
                # Handles the switch from convolutional layers to fc layers
                fc_layer0_unflattened = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                aligned_wt = torch.bmm(
                    fc_layer0_unflattened,
                    T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                ).permute(1, 2, 0)
                aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
            else:
                aligned_wt = torch.matmul(fc_layer1_weight.data, T_var)
            M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
            if args.skip_last_layer and idx == (num_layers - 1):
                if args.ensemble_step != 0.5:
                    avg_aligned_layers[layer0_name] = (1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight
                else:
                    avg_aligned_layers[layer0_name] = (aligned_wt + fc_layer1_weight) / 2
                return avg_aligned_layers

        if args.importance is None or (idx == num_layers -1):
            mu = get_histogram(args, mu_cardinality)
            nu = get_histogram(args, nu_cardinality)
        else:
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data)
            assert args.proper_marginals

        cpuM = M.data.cpu().numpy()
        # if args.exact:
        #     T = ot.emd(mu, nu, cpuM)
        # else:
        T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)

        if args.gpu_id != -1:
            T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
        else:
            T_var = torch.from_numpy(T).float()


        if args.correction:
            if not args.proper_marginals:

                if args.gpu_id != -1:
                    marginals = torch.ones(T_var.shape[0]).cuda(args.gpu_id) / T_var.shape[0]
                else:
                    marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals = torch.diag(1.0/(marginals + eps))
                T_var = torch.matmul(T_var, marginals)
            else:

                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)
                marginals = (1 / (marginals_beta + eps))
                T_var = T_var * marginals

        if args.past_correction:
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if 'adapters' in layer0_name:
            if args.ensemble_step != 0.5:
               geometric_fc = ((1 - args.ensemble_step) * t_fc0_model +
                               args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
            avg_aligned_layers[layer0_name] = geometric_fc

    for idx, (layer0_name, fc_layer0_weight) in enumerate(networks[0].named_parameters()):
        if 'adapter' in layer0_name and 'bias' in layer0_name:
            avg_aligned_layers[layer0_name] = fc_layer0_weight

    return avg_aligned_layers


def update_model(args, model, new_params, test=False, test_loader=None, idx=-1):
    updated_model = get_model_from_name(args, idx=idx)
    if args.gpu_id != -1:
        updated_model = updated_model.cuda(args.gpu_id)

    layer_idx = 0
    model_state_dict = model.state_dict()

    for key, value in model_state_dict.items():
        model_state_dict[key] = new_params[layer_idx]
        layer_idx += 1
        if layer_idx == len(new_params):
            break


    updated_model.load_state_dict(model_state_dict)

    if test:
        log_dict = {}
        log_dict['test_losses'] = []
        final_acc = routines.test(args, updated_model, test_loader, log_dict)
        print("accuracy after update is ", final_acc)
    else:
         final_acc = None

    return updated_model, final_acc


def _check_activation_sizes(args, acts0, acts1):
    if args.width_ratio == 1:
        return acts0.shape == acts1.shape
    else:
        return acts0.shape[-1]/acts1.shape[-1] == args.width_ratio


def process_activations(activations, layer0_name, layer1_name):
    activations_0 = activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')]
    activations_1 = activations[1][layer1_name.replace('.' + layer1_name.split('.')[-1], '')]

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1


def _reduce_layer_name(layer_name):

    return layer_name.replace('.' + layer_name.split('.')[-1], '')


def _get_layer_weights(layer_weight):

    return layer_weight.data


def _process_ground_metric_from_acts(ground_metric_object, activations):

    M0 = ground_metric_object.process(activations[0], activations[0])
    M1 = ground_metric_object.process(activations[1], activations[1])
    return M0, M1


def _custom_sinkhorn(args, mu, nu, cpuM):

    if not args.unbalanced:

        if args.sinkhorn_type == 'normal':
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'stabilized':
            T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'epsilon':
            T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'gpu':
            T, _ = utils.sinkhorn_loss(cpuM, mu, nu, gpu_id=args.gpu_id, epsilon=args.reg, return_tmap=True)
        else:
            raise NotImplementedError
    else:
        T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=args.reg, reg_m=args.reg_m)
    return T


def _sanity_check_tmap(T):
    if not math.isclose(np.sum(T), 1.0, abs_tol=1e-5):
        raise Exception('NAN inside Transport MAP. Most likely due to large ground metric values')

def _get_updated_acts_v0(args, layer_shape, aligned_wt, model0_aligned_layers, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    if layer_shape != aligned_wt.shape:
        updated_aligned_wt = aligned_wt.view(layer_shape)
    else:
        updated_aligned_wt = aligned_wt

    updated_model0, _ = update_model(args, networks[0], model0_aligned_layers + [updated_aligned_wt], test=True,
                                     test_loader=test_loader, idx=0)
    updated_activations = utils.get_model_activations(args, [updated_model0, networks[1]],
                                                      config=args.config,
                                                      layer_name=_reduce_layer_name(layer_names[0]), selective=True)

    updated_activations_0, updated_activations_1 = process_activations(updated_activations,
                                                                       layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1

def _get_updated_acts_v1(args, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    updated_activations = utils.get_model_activations(args, networks,
                                                      config=args.config)

    updated_activations_0, updated_activations_1 = process_activations(updated_activations,
                                                                       layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1


def _check_layer_sizes(args, layer_idx, shape1, shape2, num_layers):

    if args.width_ratio == 1:
        return shape1 == shape2
    else:
        if args.dataset == 'mnist':
            if layer_idx == 0:
                return shape1[-1] == shape2[-1] and (shape1[0]/shape2[0]) == args.width_ratio
            elif layer_idx == (num_layers -1):
                return (shape1[-1]/shape2[-1]) == args.width_ratio and shape1[0] == shape2[0]
            else:
                ans = True
                for ix in range(len(shape1)):
                    ans = ans and shape1[ix]/shape2[ix] == args.width_ratio
                return ans
        elif args.dataset[0:7] == 'Cifar10':
            assert args.second_model_name is not None
            if layer_idx == 0 or layer_idx == (num_layers -1):
                return shape1 == shape2
            else:
                if (not args.reverse and layer_idx == (num_layers-2)) or (args.reverse and layer_idx == 1):
                    return (shape1[1] / shape2[1]) == args.width_ratio
                else:
                    return (shape1[0]/shape2[0]) == args.width_ratio


def _compute_marginals(args, T_var, device=torch.device("cpu"), eps=1e-9):

    marginals = (T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).unsqueeze(1)

    marginals = (1 / (marginals + eps))
    T_var = T_var * marginals

    # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
    # this should all be ones, and number equal to number of neurons in 2nd model
    # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

    return T_var, marginals

def _get_current_layer_transport_map(args, mu, nu, M0, M1, idx, layer_name=None):

    if not args.gromov:
        cpuM = M0.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = _custom_sinkhorn(args, mu, nu, cpuM)

    else:
        cpuM0 = M0.data.cpu().numpy()
        cpuM1 = M1.data.cpu().numpy()

        assert not args.exact
        T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=args.gromov_loss, epsilon=args.reg)

    if not args.unbalanced:
        _sanity_check_tmap(T)

    if args.gpu_id != -1:
        T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
    else:
        T_var = torch.from_numpy(T).float()
    return T_var

def _get_neuron_importance_histogram(args, layer_weight, is_conv=False, eps=1e-9):

    if is_conv:
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()
    
    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
                    np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
                    np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist/importance_hist.sum())

    return importance_hist


def get_acts_wassersteinized_layers_modularized(args, networks, activations, skip_layers=[], eps=1e-7):

    '''
    Average based on the activation vector over data samples. Obtain the transport map,
    and then based on which align the nodes and average the weights!
    Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''


    avg_aligned_layers = {}
    T_var = None
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    ground_metric_object = GroundMetric(args)

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    incoming_layer_aligned = True

    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

        # todo skip the embedding layers
        skip = False
        for layer in skip_layers:
            if layer in layer0_name:
                skip = True
                break
        if skip:
            continue
        assert fc_layer0_weight.shape == fc_layer1_weight.shape

        activations_0, activations_1 = process_activations(activations, layer0_name, layer1_name)

        assert activations_0.shape[0] == fc_layer0_weight.shape[0]
        assert activations_1.shape[0] == fc_layer1_weight.shape[0]
        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        fc_layer0_weight_data = _get_layer_weights(fc_layer0_weight)
        fc_layer1_weight_data = _get_layer_weights(fc_layer1_weight)

        if idx == 0 or incoming_layer_aligned:
            aligned_wt = fc_layer0_weight_data

        else:
            if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                   -1).permute(2, 0, 1)
                aligned_wt = torch.bmm(
                    fc_layer0_unflattened,
                    T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                ).permute(1, 2, 0)
                aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
            else:
                aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)

        mu = get_histogram(args,  mu_cardinality)
        nu = get_histogram(args,  nu_cardinality)
        M0, M1 = _process_ground_metric_from_acts(ground_metric_object, [activations_0, activations_1])
        if args.skip_last_layer and idx == (num_layers - 1):

            if args.skip_last_layer_type == 'average':
                if args.ensemble_step != 0.5:
                    avg_aligned_layers[layer0_name] = (1 - args.ensemble_step) * aligned_wt \
                                                      + args.ensemble_step * fc_layer1_weight
                else:
                    avg_aligned_layers[layer0_name] = (aligned_wt + fc_layer1_weight) / 2
            elif args.skip_last_layer_type == 'second':
                avg_aligned_layers[layer0_name] = fc_layer1_weight

            return avg_aligned_layers

        T_var = _get_current_layer_transport_map(args, mu, nu, M0, M1, idx=idx, layer_name=layer0_name)
        T_var, marginals = _compute_marginals(args, T_var, device, eps=eps)
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())
        if args.past_correction:
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if 'adapters' in layer0_name:
            if args.ensemble_step != 0.5:
                geometric_fc = (1 - args.ensemble_step) * t_fc0_model + \
                               args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            else:
                geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
            avg_aligned_layers[layer0_name] = geometric_fc
        incoming_layer_aligned = False
        idx += 1

    for idx, (layer0_name, fc_layer0_weight) in enumerate(networks[0].named_parameters()):
        if 'adapter' in layer0_name and 'bias' in layer0_name:
            avg_aligned_layers[layer0_name] = fc_layer0_weight

    return avg_aligned_layers


def get_network_from_param_list(args, param_list, new_network):

    # new_network = get_model_from_name(args, idx=1)
    if args.gpu_id != -1:
        new_network = new_network.cuda(args.gpu_id)

    # check the test performance of the network before
    acc = None
    assert len(list(new_network.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_network.state_dict()

    for key, value in model_state_dict.items():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1

    new_network.load_state_dict(model_state_dict)

    # check the test performance of the network after
    log_dict = {}
    log_dict['test_losses'] = []

    return acc, new_network


skip_layers = ['.embeddings', '.bias', '.LayerNorm', '.pooler', 'classifier']

def geometric_ensembling_modularized(args, networks, activations=None, skip_layers=skip_layers):
    avg_aligned_layers = None
    if args.geom_ensemble_type == 'wts':
        avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks, activations, skip_layers)
    elif args.geom_ensemble_type == 'acts':
        avg_aligned_layers = get_acts_wassersteinized_layers_modularized(args, networks, activations, skip_layers)
    return avg_aligned_layers

