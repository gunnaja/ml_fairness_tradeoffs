from nsga2 import nsga2
from population import get_C, get_gamma, get_selected_features
from metrics import function_name_to_string
from algorithms import svm_reweighing
from filehandler import write_result_to_file

FITNESS_SCORES = {}


def svm_reweighing_experiment(num_generations, population_size, mutation_rate, crossover_rate, chromosome_length,
                              fairness_metric, accuracy_metric, training_data, test_data, privileged_groups,
                              unprivileged_groups, max_iter, svm_seed):

    def evaluation_function(chromosome):
        if str(chromosome) in FITNESS_SCORES:
            return FITNESS_SCORES[str(chromosome)]
        else:
            C = get_C(chromosome)
            gamma = get_gamma(chromosome)
            selected_features = get_selected_features(chromosome, 30)
            accuracy_score, fairness_score = svm_reweighing(training_data=training_data, test_data=test_data,
                                                            fairness_metric=fairness_metric,
                                                            accuracy_metric=accuracy_metric,
                                                            C=C, gamma=gamma, keep_features=selected_features,
                                                            privileged_groups=privileged_groups,
                                                            unprivileged_groups=unprivileged_groups, max_iter=max_iter,
                                                            svm_seed=svm_seed)
            FITNESS_SCORES[str(chromosome)] = [accuracy_score, fairness_score]
            return [accuracy_score, fairness_score]

    result = nsga2(pop_size=population_size,
                   num_generations=num_generations,
                   chromosome_length=chromosome_length,
                   crossover_rate=crossover_rate,
                   mutation_rate=mutation_rate,
                   evaluation_algorithm=evaluation_function)

    result_summary = {'name': 'SVM_Reweighing_',
                      'result': result,
                      'fairness_metric': function_name_to_string(fairness_metric),
                      'accuracy_metric': function_name_to_string(accuracy_metric),
                      'nsga2_parameters': {'num_generations': num_generations, 'population_size': population_size,
                                           'crossover_rate': crossover_rate, 'mutation_rate': mutation_rate,
                                           'chromosome_length': chromosome_length}}
    write_result_to_file(result_summary, "svm_reweighing")

    return result
