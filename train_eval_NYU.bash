folder="nyu_experiment"

ex1="python main.py --check $folder --config_file configs/nyu.yaml"
ex2="python eval.py --path $folder/checkpoints"

$ex1
$ex2
