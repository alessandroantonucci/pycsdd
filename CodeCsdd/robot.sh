for ess in 1; do
  for training_size in 100; do
    for testing_size in 2; do
      for prob in 0.05 0.1 0.2 0.3 0.4; do
        echo "python experiment_seven_segments_no_digits.py $ess $training_size $testing_size $prob"
        python experiment_seven_segments_no_digits.py $ess $training_size $testing_size $prob
      done
    done
  done
done
