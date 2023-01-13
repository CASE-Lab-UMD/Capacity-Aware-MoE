import copy


def bert_test(models, dataloader):

    for model in models:
        model.eval()
    total = 0
    true_nums = 0
    total_steps = len(dataloader)
    for i, input in enumerate(dataloader):
        if 'idx' in input.keys():
            del input['idx']
        logits = 0    
        for model in models:
            output = model(**input)
            logits = output['logits'] + logits
        predicts = logits.argmax(axis=1)        
        total += len(input['labels'])

        true_nums += (predicts == input['labels']).sum().item()
        if (i+1) % 10 == 0:
            print(f"[{i}/{total_steps}]")
    accuracy = true_nums/total
    print(f"accuracy: {true_nums}/{total}={accuracy}")
    return accuracy

def bert_test2(models, dataloader):

    total = 0
    true_nums = 0
    total_steps = len(dataloader)
    for i, input in enumerate(dataloader):
        if 'idx' in input.keys():
            del input['idx']
        bert_input = copy.deepcopy(input)
        if 'labels' in bert_input.keys():
            del bert_input['labels']

        logits = 0 
        pooled_output = 0   
        for model in models:
            output = model.bert(**bert_input)
            pooled_output += output[1]
        pooled_output /= len(models)
        logits = models[0].classifier(pooled_output) 

        predicts = logits.argmax(axis=1)        
        total += len(input['labels'])

        true_nums += (predicts == input['labels']).sum().item()
        if (i+1) % 10 == 0:
            print(f"[{i}/{total_steps}]")
    accuracy = true_nums/total
    print(f"accuracy: {true_nums}/{total}={accuracy}")
    return accuracy