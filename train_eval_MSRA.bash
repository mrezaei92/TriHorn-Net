folder_results="msra_experiment"
main_predFileName="main_pred.txt"

mkdir -p $folder_results

for f in $(seq 0 8)
do
   python main.py --check MSRA_P${f} --config_file configs/msra.yaml --leaveout_subject $f
   python eval.py --path MSRA_P${f}/checkpoints --save_preds best --pred_file_name preds_${f}.txt
   mv log.txt MSRA_P${f}
   mv results.txt MSRA_P${f}
   
   mv MSRA_P${f} $folder_results/MSRA_P${f}
   echo "Experiment Subject $f Finished at : $(date)">>progress.txt
   
   cat $folder_results/MSRA_P${f}/preds_${f}.txt>>$main_predFileName
  
done

echo >>progress.txt
echo "-------------------------- Aggregate Result --------------" >>progress.txt
python utils/compute3Derror_MSRA.py $main_predFileName >>progress.txt
mv progress.txt $folder_results
mv $main_predFileName $folder_results

