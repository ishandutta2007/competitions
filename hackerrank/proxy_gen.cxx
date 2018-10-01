#include "common/proxy_run_base.h"
#include "proxy_gen.h"

TSolutionsMap solutions_map {
	{ "base", main_base },
	{ "cube_summation", main_cube_summation },
	{ "direct_connections__bit_sparse", main_direct_connections__bit_sparse },
	{ "direct_connections__coordinate_compression", main_direct_connections__coordinate_compression },
	{ "direct_connections__treap", main_direct_connections__treap },
	{ "jim_and_the_skyscrapers", main_jim_and_the_skyscrapers },
	{ "kindergarten_adventures", main_kindergarten_adventures },
	{ "kindergarten_adventures__bit", main_kindergarten_adventures__bit },
	{ "mr_x_and_his_shots", main_mr_x_and_his_shots },
	{ "palindromic_subsets__segment_tree", main_palindromic_subsets__segment_tree },
	{ "palindromic_subsets__treap", main_palindromic_subsets__treap },
	{ "polynomial_division", main_polynomial_division },
	{ "self_driving_bus", main_self_driving_bus },
	{ "subsequence_weighting__bit_sparse", main_subsequence_weighting__bit_sparse },
	{ "subsequence_weighting__segment_tree", main_subsequence_weighting__segment_tree },
	{ "subsequence_weighting__treap", main_subsequence_weighting__treap },
	{ "2d_arrays_ds", main_2d_arrays_ds },
	{ "array_manipulation", main_array_manipulation },
	{ "arrays_ds", main_arrays_ds },
	{ "dynamic_array", main_dynamic_array },
	{ "left_rotation", main_left_rotation },
	{ "sparse_arrays", main_sparse_arrays },
	{ "array_and_simple_queries__splay", main_array_and_simple_queries__splay },
	{ "array_and_simple_queries__treap", main_array_and_simple_queries__treap },
	{ "median_updates__fixed_tree", main_median_updates__fixed_tree },
	{ "median_updates__upf_red_black", main_median_updates__upf_red_black },
	{ "median_updates__upf_scapegoat", main_median_updates__upf_scapegoat },
	{ "median_updates__upf_treap", main_median_updates__upf_treap },
	{ "median_updates__upt_red_black", main_median_updates__upt_red_black },
	{ "median_updates__upt_scapegoat", main_median_updates__upt_scapegoat },
	{ "median_updates__upt_splay", main_median_updates__upt_splay },
	{ "median_updates__upt_treap", main_median_updates__upt_treap },
	{ "components_in_a_graph", main_components_in_a_graph },
	{ "kundu_and_tree", main_kundu_and_tree },
	{ "merging_communities", main_merging_communities },
	{ "super_maximum_cost_queries", main_super_maximum_cost_queries },
	{ "find_the_running_median", main_find_the_running_median },
	{ "jesse_and_cookies", main_jesse_and_cookies },
	{ "minimum_average_waiting_time", main_minimum_average_waiting_time },
	{ "qheap1", main_qheap1 },
	{ "castle_on_the_grid", main_castle_on_the_grid },
	{ "down_to_zero_ii", main_down_to_zero_ii },
	{ "queries_with_fixed_length", main_queries_with_fixed_length },
	{ "queue_using_two_stacks", main_queue_using_two_stacks },
	{ "truck_tour", main_truck_tour },
	{ "and_xor_or", main_and_xor_or },
	{ "balanced_brackets", main_balanced_brackets },
	{ "equal_stacks", main_equal_stacks },
	{ "game_of_two_stacks", main_game_of_two_stacks },
	{ "largest_rectangle", main_largest_rectangle },
	{ "maximum_element", main_maximum_element },
	{ "poisonous_plants", main_poisonous_plants },
	{ "simple_text_editor", main_simple_text_editor },
	{ "waiter", main_waiter },
	{ "balanced_forest", main_balanced_forest },
	{ "jennys_subtrees", main_jennys_subtrees },
	{ "kittys_calculations_on_a_tree", main_kittys_calculations_on_a_tree },
	{ "swap_nodes_algo", main_swap_nodes_algo },
	{ "tree_coordinates", main_tree_coordinates },
	{ "contacts", main_contacts },
	{ "no_prefix_set", main_no_prefix_set },
	{ "army_game", main_army_game },
	{ "best_divisor", main_best_divisor },
	{ "bus_station", main_bus_station },
	{ "connecting_towns", main_connecting_towns },
	{ "cutting_paper_squares", main_cutting_paper_squares },
	{ "die_hard_3", main_die_hard_3 },
	{ "diwali_lights", main_diwali_lights },
	{ "even_odd_query", main_even_odd_query },
	{ "filling_jars", main_filling_jars },
	{ "find_the_point", main_find_the_point },
	{ "halloween_party", main_halloween_party },
	{ "handshake", main_handshake },
	{ "is_fibo", main_is_fibo },
	{ "jim_and_the_jokes", main_jim_and_the_jokes },
	{ "k_candy_store", main_k_candy_store },
	{ "leonardos_prime_factors", main_leonardos_prime_factors },
	{ "matrix_tracing", main_matrix_tracing },
	{ "maximum_draws", main_maximum_draws },
	{ "minimum_height_triangle", main_minimum_height_triangle },
	{ "most_distant", main_most_distant },
	{ "mutual_recurrences", main_mutual_recurrences },
	{ "possible_path", main_possible_path },
	{ "restaurant", main_restaurant },
	{ "reverse_game", main_reverse_game },
	{ "russian_peasant_exponentiation", main_russian_peasant_exponentiation },
	{ "sherlock_and_divisors", main_sherlock_and_divisors },
	{ "sherlock_and_moving_tiles", main_sherlock_and_moving_tiles },
	{ "sherlock_and_permutations", main_sherlock_and_permutations },
	{ "special_multiple", main_special_multiple },
	{ "strange_grid_again", main_strange_grid_again },
	{ "sumar_and_the_floating_rocks", main_sumar_and_the_floating_rocks },
	{ "summing_the_n_series", main_summing_the_n_series },
	{ "akhil_and_gf", main_akhil_and_gf },
	{ "bear_and_cryptography", main_bear_and_cryptography },
	{ "binomial_coefficients", main_binomial_coefficients },
	{ "breaking_sticks", main_breaking_sticks },
	{ "cheese_and_random_toppings", main_cheese_and_random_toppings },
	{ "closest_number", main_closest_number },
	{ "constructing_a_number", main_constructing_a_number },
	{ "dancing_in_pairs", main_dancing_in_pairs },
	{ "divisor_exploration", main_divisor_exploration },
	{ "divisor_exploration_ii", main_divisor_exploration_ii },
	{ "easy_gcd", main_easy_gcd },
	{ "equations", main_equations },
	{ "eulers_criterion", main_eulers_criterion },
	{ "fibonacci_finding_easy", main_fibonacci_finding_easy },
	{ "fibonacci_gcd", main_fibonacci_gcd },
	{ "help_mike", main_help_mike },
	{ "identify_smith_numbers", main_identify_smith_numbers },
	{ "john_and_gcd_list", main_john_and_gcd_list },
	{ "largest_non_coprime_submatrix", main_largest_non_coprime_submatrix },
	{ "little_ashishs_huge_donation", main_little_ashishs_huge_donation },
	{ "little_panda_power", main_little_panda_power },
	{ "lucy_and_flowers", main_lucy_and_flowers },
	{ "manasa_and_factorials", main_manasa_and_factorials },
	{ "manasa_loves_maths", main_manasa_loves_maths },
	{ "mehta_and_his_laziness", main_mehta_and_his_laziness },
	{ "number_of_zero_xor_subsets", main_number_of_zero_xor_subsets },
	{ "power_of_large_numbers", main_power_of_large_numbers },
	{ "primitive_problem", main_primitive_problem },
	{ "salary_blues", main_salary_blues },
	{ "satisfactory_pairs", main_satisfactory_pairs },
	{ "scalar_products", main_scalar_products },
	{ "sherlock_and_gcd", main_sherlock_and_gcd },
	{ "superpowers_of_2", main_superpowers_of_2 },
	{ "the_chosen_one", main_the_chosen_one },
	{ "the_chosen_one__treap", main_the_chosen_one__treap },
	{ "twins", main_twins },
};
