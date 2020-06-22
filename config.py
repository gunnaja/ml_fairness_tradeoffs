from metrics import auc, average_odds_difference, statistical_parity_difference, theil_index, equal_opportunity_difference, binary_accuracy, disparate_impact

theil_config = {
  "num_generations": 100,
  "pop_size": 50,
  "mutation_rate": 0.05,
  "crossover_rate": 0.7,
  "svm_max_iter": 10000,
  "svm_seed": 0,
  "fairness_metric": theil_index,
  "accuracy_metric": binary_accuracy
}

statistical_parity_config = {
  "num_generations": 100,
  "pop_size": 50,
  "mutation_rate": 0.05,
  "crossover_rate": 0.7,
  "svm_max_iter": 1000000,
  "svm_seed": 0,
  "fairness_metric": statistical_parity_difference,
  "accuracy_metric": binary_accuracy
}

equal_opportunity_config = {
  "num_generations": 100,
  "pop_size": 50,
  "mutation_rate": 0.05,
  "crossover_rate": 0.7,
  "svm_max_iter": 1000000,
  "svm_seed": 0,
  "fairness_metric": equal_opportunity_difference,
  "accuracy_metric": binary_accuracy
}

disparate_impact_config = {
  "num_generations": 100,
  "pop_size": 50,
  "mutation_rate": 0.05,
  "crossover_rate": 0.7,
  "svm_max_iter": 15000,
  "svm_seed": 0,
  "fairness_metric": disparate_impact,
  "accuracy_metric": binary_accuracy
}

average_odds_config = {
  "num_generations": 100,
  "pop_size": 50,
  "mutation_rate": 0.05,
  "crossover_rate": 0.7,
  "svm_max_iter": 15000,
  "svm_seed": 0,
  "fairness_metric": average_odds_difference,
  "accuracy_metric": binary_accuracy
}