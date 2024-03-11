import wandb
api = wandb.Api()

import matplotlib.pyplot as plt

test_obj = ['mtndew', 'pepsi', 'obamna', 'cola']

def run_to_obj_mean_rew(run_name):
    run = api.run(run_name)
    full_history = run._full_history()
    n_test = 5
    test_start = 100000
    obj_to_rew_hist = {}
    eval_steps = []
    for obj in test_obj:
        obj_to_rew_hist[obj] = []
    for hist_i, history in enumerate(full_history):
        if 'test/mean_score' in history.keys():
            eval_steps.append(hist_i)
            obj_to_rews = {}
            for obj in test_obj:
                obj_to_rews[obj] = []
            for key in history.keys():
                if key.split('/')[0] == 'test' and key.split('_')[-1] in test_obj and key.split('_')[1] == 'max':
                    obj_to_rews[key.split('_')[-1]].append(history[key])
            for obj in test_obj:
                obj_to_rew_hist[obj].append(sum(obj_to_rews[obj])/n_test)
    return obj_to_rew_hist, eval_steps

obj_to_rew_hist_1, eval_steps_1 = run_to_obj_mean_rew('uiuc-yixuan/diffusion_policy_debug/2e46ec5w') # 2023.11.08_sapien_can_d3fields_middle_view_demo_100_pn_comp_pos
name_1 = '2023.11.08_sapien_can_d3fields_middle_view_demo_100_pn_comp_pos'
obj_to_rew_hist_2, eval_steps_2 = run_to_obj_mean_rew('uiuc-yixuan/diffusion_policy_debug/1mz2rjxr') # 2023.11.08_sapien_can_d3fields_middle_view_demo_100_no_feats_pn_comp_pos
name_2 = '2023.11.08_sapien_can_d3fields_middle_view_demo_100_no_feats_pn_comp_pos'
obj_to_rew_hist_3, eval_steps_3 = run_to_obj_mean_rew('uiuc-yixuan/diffusion_policy_debug/dkeqnuxb') # 2023.11.08_sapien_can_d3fields_middle_view_demo_100_no_feats_pn
name_3 = '2023.11.08_sapien_can_d3fields_middle_view_demo_100_no_feats_pn'

for obj in test_obj:
    plt.figure()
    plt.plot(eval_steps_1, obj_to_rew_hist_1[obj], label=name_1)
    plt.plot(eval_steps_2, obj_to_rew_hist_2[obj], label=name_2)
    plt.plot(eval_steps_3, obj_to_rew_hist_3[obj], label=name_3)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.title(obj)
    plt.show()
    plt.close()

