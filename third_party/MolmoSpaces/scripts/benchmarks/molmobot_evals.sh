# Mini-Bench results.
python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125125_pick_msproc_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_045917 palig --success-condition oracle --output-csv data/pick_easy/palig.csv 
python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125201_pick_msproc_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050236 pi05ft --success-condition oracle  --output-csv data/pick_easy/pi05ft.csv
python eval_to_csv.py /weka/prior/datasets/robomolmo/eval_results/eval_pick_full_mi-Frnk-8n_abs_vid_2f_8gap_2p-03-08-18-52-01_bs1024_dbs16_stp50000-mix_5_feb20-step50000_20260310183024 molmobot_f2 --success-condition oracle  --output-csv data/pick_easy/molmobot_f2.csv
python eval_to_csv.py /weka/prior/datasets/robomolmo/eval_results/eval_pick_full_mi-Frnk-16n_abs_vid_3f_8gap_2p-03-08-22-40-23_bs1024_dbs8_stp50000-mix_5_feb20-step30000_20260310173326 molmobot_f3 --success-condition oracle  --output-csv data/pick_easy/molmobot_f3t.csv
python eval_to_csv.py /weka/prior/datasets/robomolmo/eval_results/dist-eval-Frnk-8n_abs_-03-06-17-32-00_bs1024_dbs16_stp200000-mix_5_feb20-step200000_20260309134451 molmobot_image --success-condition oracle  --output-csv data/pick_easy/molmobot_imag.csv


magic-wormhole send /root/code/mujoco-thor/scripts/benchmarks/data/pick_easy



# python eval_to_csv.py /root/code/mujoco-thor/eval_output/mujoco_thor.evaluation.configs.evaluation_configs:PiPolicyEvalConfig/20260311_014816 pi05 --output-csv pi05_pick_cls.csv
# python eval_to_csv.py /root/code/mujoco-thor/eval_output/mujoco_thor.evaluation.configs.evaluation_configs:PiPolicyEvalConfig/20260309_221415 pi05 --output-csv pi05_pick_fil.csv
# python eval_to_csv.py /root/code/mujoco-thor/eval_output/mujoco_thor.evaluation.configs.evaluation_configs:PiPolicyEvalConfig/20260309_233526/ pi05 --output-csv pi05_pick_rnd.csv
# python eval_to_csv.py /weka/prior/datasets/robomolmo/eval_results/dist-eval-checkpoints-pi05_droid_jointpos_20260311051846 pi05 --output-csv pi05_pnp.csv
# python eval_to_csv.py /weka/prior/datasets/robomolmo/eval_results/dist-eval-checkpoints-pi05_droid_jointpos_20260311042056 pi05 --output-csv pi05_pnp_nt.csv
# python eval_to_csv.py /weka/prior/datasets/robomolmo/eval_results/dist-eval-checkpoints-pi05_droid_jointpos_20260311044757 pi05 --output-csv pi05_pnp_cl.csv
# zip pi05_results.zip pi05_pick_cls.csv pi05_pick_fil.csv pi05_pick_rnd.csv pi05_pnp.csv pi05_pnp_nt.csv pi05_pnp_cl.csv

# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125125_pick_msproc_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_045917 palig --output-csv data/pick_easy/palig.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125131_pick_classic_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_045926 palig --output-csv palig_pick_cls.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125136_pick_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_050033 palig --output-csv palig_pick_fil.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125141_pick_randomcam_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_050018 palig --output-csv palig_pick_rnd.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125146_pnp_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_050025 palig --output-csv palig_pnp.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125151_pnp_nextto_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_050030 palig --output-csv palig_pnp_nt.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125156_pnp_color_paligemma/eval_data/pi/200000/PiPnPBenchmarkEvalConfig/20260311_050039 palig --output-csv palig_pnp_cl.csv
# zip palig_results.zip palig_pick_msproc.csv palig_pick_cls.csv palig_pick_fil.csv palig_pick_rnd.csv palig_pnp.csv palig_pnp_nt.csv palig_pnp_cl.csv

# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125201_pick_msproc_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050236 pi05ft --output-csv data/pick_easy/pi05ft.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125206_pick_classic_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050247 pi05ft --output-csv pi05ft_pick_cls.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125211_pick_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050144 pi05ft --output-csv pi05ft_pick_fil.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125217_pick_randomcam_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050145 pi05ft --output-csv pi05ft_pick_rnd.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125222_pnp_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050048 pi05ft --output-csv pi05ft_pnp.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125227_pnp_nextto_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050123 pi05ft --output-csv pi05ft_pnp_nt.csv
# python eval_to_csv.py /weka/prior/abhayd/sim_cotraining_output/eval_runs_openpi/20260311_125232_pnp_color_pi05ft/eval_data/pi/15000/PiPnPBenchmarkEvalConfig/20260311_050158 pi05ft --output-csv pi05ft_pnp_cl.csv
# zip pi05ft_results.zip pi05ft_pick_msproc.csv pi05ft_pick_cls.csv pi05ft_pick_fil.csv pi05ft_pick_rnd.csv pi05ft_pnp.csv pi05ft_pnp_nt.csv pi05ft_pnp_cl.csv


