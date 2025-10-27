import os
import json
import pandas as pd

capacity = 2.0
# samples = 10000
# dir_name = f"/workspace/MoD/results/trained_models/mixed/{samples}"
# model = "qwen-2.5-3b-mod"
# model = "qwen-2.5-1.5b-mod"

model = "mistral-7b-mod"
# model = "Meta-Llama-3-8B"

# root_name = f"/workspace/MoD/results/trained_models/{model}/mixed"

root_name = f"/workspace/models/deepseek-moe-16b-base-temp/expert_capacity-{capacity}"
root_name = f"/workspace/models/OLMoE-1B-7B-temp/expert_capacity-{capacity}"

# root_name = "/workspace/models/deepseek-moe-16b-base"

acc_dict = {
    "arc_challenge": "acc_norm,none", 
    "boolq": "acc,none", 
    "gsm8k": "exact_match,flexible-extract", 
    "hellaswag": "acc_norm,none", 
    "mmlu": "acc,none", 
    "openbookqa": "acc_norm,none", 
    "piqa": "acc_norm,none", 
    "rte": "acc,none", 
    "winogrande": "acc,none", 
}

task_name_dict = {
    "arc_challenge": "ARC-C", 
    "boolq": "BoolQ", 
    "gsm8k": "GSM8K", 
    "hellaswag": "HellaSwag", 
    "mmlu": "MMLU", 
    "openbookqa": "OBQA", 
    "piqa": "PIQA", 
    "rte": "RTE", 
    "winogrande": "WinoGrande", 
}
# Create a list to store results for each experiment
results = []
for path in os.listdir(root_name):
    
    exp = f"{path}"
    # exp = exp.replace("attn_sequence_epoch1_", "").replace("gradient_scale", "gs")
    path = os.path.join(root_name, path)
    if not os.path.isdir(path):
        continue
    # print(path)
    avg = 0.
    task_results = {"Exp": exp}  # Dictionary to store individual task results
    
    for task in os.listdir(path): 
        if task.endswith(".json") and not os.path.isfile(os.path.join(path, task)):
            path_task = os.path.join(path, task)
            task_name = task.split('.')[0]
            for file in os.listdir(path_task):
    
                file = os.path.join(path_task, file)

                # try:
                with open(file, 'r') as file:
                    data = json.load(file)
                # except:
                #     continue
                result = data["results"][task_name]
                # for key in result:
                task = task.split(".")[0]
                # print(task, acc_dict[task_name], result[acc_dict[task_name]])
                # print(exp, task, result[acc_dict[task_name]])
                
                # Store the task result in the dictionary
                if task_name in task_results: 
                    avg -= task_results[task_name]
                new_task_name = task_name_dict[task.split('.')[0]]
                task_results[new_task_name] = round(result[acc_dict[task_name]], 3)
                avg += task_results[new_task_name]
                
                # print(task, result[acc_dict[task_name]])
                # print(avg)
                
    task_results["Avg."] = round(avg / len(acc_dict), 3)
    if task_results["Avg."] > 0: 
        results.append(task_results)
    # print(task_results["avg_performance"])
                
    # print(f"avg performance: {avg / len(acc_dict)}")
    # print()

    # break

    
# Convert the list of results to a pandas DataFrame
df = pd.DataFrame(results)


# Save the DataFrame to an Excel file
output_file = f"{root_name}/output.xlsx"
df.to_excel(output_file, index=False)

print(f"Results have been saved to {output_file}")
