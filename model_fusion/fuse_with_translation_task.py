import sys
sys.path.append('/user/sunsiqi/hs/ModelFusion/otfusion')
import utils_zz as uz
import utils_zz2 as uz2
import utils
import activation_utils as au
import prepare
import validates
import copy
import fairseq
fairseq.modules.MultiheadAttention.functional_self_attention = False
import torch


if __name__ == '__main__':
    import deberta_prepare as dprepare
    exp=sys.argv[1]
    aux_task = sys.argv[2] # RI
    tgt_task = sys.argv[3] # FT target main model 
    pend = sys.argv[4]
    p0 = sys.argv[5] # percentage of aux model
    
    # tag = 'bert-base-uncased'
    # tag = "deberta-base"
    tag = 'mBART'

    if tag == "deberta-base":
        prepare_module = dprepare
    else:
        prepare_module = prepare
    
    args, net1, net2, aux_inputs, tgt_inputs = prepare_module.mbart_translation_test(exp=exp, aux_task=aux_task, tgt_task=tgt_task)
    # TODO glue_ft_test 
    # args.gpu_id = int(sys.argv[6])
    
    acts_inputs = aux_inputs
    v2_activationManager = au.ActivationsManager(args, [copy.deepcopy(net1), copy.deepcopy(net2)], acts_inputs)
    # v2_activationManager = au.ActivationsManager(args, [copy.deepcopy(net1), copy.deepcopy(net2)], [aux_inputs, tgt_inputs])

    # args.gpu_id = 0
    args.mem_efficient = True
    # v2_avged_model, v2_aligned_model0, v2_actual_activations_T, v2_am = uz2.adaptive_fuse_model2(args, net1, net2, v2_activationManager,
    #     skip_in=['bert.embeddings.word_embeddings', 'bert.embeddings.position_embeddings', 'bert.embeddings.token_type_embeddings'], skip_out=['classifier'])
    
    # 不对embedding层和layernor_embedding做融合
    v2_avged_model, v2_aligned_model0, v2_actual_activations_T, v2_am = uz2.adaptive_fuse_model2(args, net1, net2, v2_activationManager,
        skip_in=['encoder.embed_tokens', 'encoder.embed_positions', 'decoder.embed_tokens', 'decoder.embed_positions', 'decoder.layernorm_embedding', 'encoder.layernorm_embedding'], \
            skip_out=['encoder.embed_tokens', 'encoder.embed_positions', 'decoder.embed_tokens', 'decoder.embed_positions', 'decoder.output_projection'], p0=float(p0))
    # v2_avged_model = p0 * net1(aux) + ( 1 - p0 ) * net2 (tgt)
    # TODO skip_in;skip_out
    print(1)
    torch.save( v2_avged_model.state_dict(), \
        '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoints/fusioned_state/{}_FT_main.pt'.format(pend))
    
    # torch.save( v2_aligned_model0.state_dict(), \
    #     '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoints/fusioned_state/{}.pt'.format(pend))
    # torch.save( v2_avged_model.state_dict(), \
    #     '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoints/fusioned_state/enzh_RI_avged_model{}.pt'.format(pend))
    # torch.save( v2_aligned_model0.state_dict(), \
    #     '/workspace/data/users/zanchangtong1/2_High_Resource_Translation/checkpoints/fusioned_state/enzh_RI_aligned_model{}.pt'.format(pend))
    # validates.bert_test([v2_avged_model], eval_dataloader1)  # 0.6838235294117647
    # validates.bert_test([v2_aligned_model0], eval_dataloader1)  # 0.8406862745098039
    # validates.bert_test([net1], eval_dataloader1)  # 0.8406862745098039
    # validates.bert_test([net2], eval_dataloader1)   # 0.8406862745098039
    # prepare_module.save_model(v2_avged_model, tgt_task, tag, f"avg_with_{aux_task}_v1.bin") 
    # prepare_module.save_model(v2_aligned_model0, tgt_task, tag, f"align_{aux_task}_v1.bin") 
    # TODO bert_test; save_model
