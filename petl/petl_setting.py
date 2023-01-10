import sys
sys.path = ['/user/sunsiqi/hs/MoE', '/user/sunsiqi/hs/MoE/transformers',
            '/user/sunsiqi/.pycharm_helpers/pydev', '/user/sunsiqi/.pycharm_helpers/pycharm_display', '/user/sunsiqi/.pycharm_helpers/third_party/thriftpy',
            '/Users/Lenovo/AppData/Local/JetBrains/PyCharm2021.2/cythonExtensions', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python38.zip', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8',
            '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/lib-dynload', '/user/sunsiqi/.local/lib/python3.8/site-packages',
            '/user/sunsiqi/.local/lib/python3.8/site-packages/pdbx-1.0-py3.8.egg', '/user/sunsiqi/openfold/lib/conda/envs/ddg/lib/python3.8/site-packages',
            '/user/sunsiqi/.pycharm_helpers/pycharm_matplotlib_backend']

def PETL_Setting(pretrained_model, args, logger):

    not_freeze_set, freeze_set = [], []
    all_match = False
    if args.unfreeze_params != 'none':
        not_freeze_set = args.unfreeze_params.split(',')
        freeze_set = args.freeze_set.split(',')

    logger.info(not_freeze_set)

    def check_params(module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)

    for n, p in pretrained_model.named_parameters():
        tune = False
        if len(not_freeze_set) > 0 and check_params(n, not_freeze_set, all_match=all_match):
            p.requires_grad = True
            tune = True
        else:
            p.requires_grad = False

        if len(freeze_set) > 0 and check_params(n, freeze_set, all_match=False):
            p.requires_grad = False
            tune = False
        if tune:
            print("tune " + n)

    logger.info("already freezed parameters!")