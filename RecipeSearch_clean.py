import pandas as pd
import ast  # abstract syntax tree
import openpyxl
import numpy as np
import csv
from itertools import combinations

global_recommneded_id = 0


def read_csv(path):
    return pd.read_csv(path)


def write_csv(name, data):
    with open(name, "w+", newline='') as recipe_csv:
        newarray = csv.writer(recipe_csv, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        newarray.writerows(data)


def re_arrange_recipe_db_calories_quantity(recipe_file_pd):
    calorie_value = []
    recipe_quantity_value = []
    for row in recipe_file_pd['calories']:
        res = ast.literal_eval(row)
        calorie_value.append(res['value'])

    print(len(calorie_value))
    for row_1 in recipe_file_pd['ingredients']:
        res = ast.literal_eval(row_1)
        for i, item in enumerate(res):
            recipe_quantity_value.append(item['quantity'])
            break
    print(len(recipe_quantity_value))
    calories_per_single_serving = []
    #

    for calorie, quantity in zip(calorie_value, recipe_quantity_value):
        print(calorie, quantity)
        if quantity == 0:
            quantity = 1
        calories_per_single_serving.append(calorie/ quantity)

    print(calories_per_single_serving)

    data = pd.read_csv('recipe_db_minimized.csv')
    data['calories'] = calories_per_single_serving
    data.to_csv('recipe_db_minimized_v2.csv')
    return


def re_arrange_recipe_db(recipe_file_pd):
    recipe_dict = {}
    recipe_no = 0

    for row in recipe_file_pd['nutritionalInfo']:
        res = ast.literal_eval(row)
        label_array = []
        quantity_array = []
        for i, item in enumerate(res):
            label_array.append(item['label'])
            quantity_array.append(item['quantity'])
        recipe_dict[recipe_no] = list(zip(label_array, quantity_array))
        recipe_no += 1

    recipe_db_minimized = []
    for k, v in recipe_dict.items():
        # print(k, v[0][0], v[0][1])  # Fat 25.34  # v give array of array, v[0] will give (fat, value)
        for element, quantity in v:
            if element == 'Fat':
                print(element, quantity)
                fat_quantity = quantity
            if element == 'Protein':
                print(element, quantity)
                protein_quantity = quantity
            if element == 'Carbohydrates':
                print(element, quantity)
                carb_quantity = quantity
        recipe_db_minimized.append([fat_quantity, carb_quantity, protein_quantity])

    print(recipe_db_minimized)
    # [ [fat, carbohydrate , protein], [fat, carbohydrate , protein], ...] ] array[k][m/n/o] where k~ 0 to 20511,
    # m= fat, n= carbohydrate, o= protein
    write_csv('recipe_db_minimized.csv', recipe_db_minimized)
    return recipe_db_minimized


def diet_plan_day(file_1):
    file_1_pd = read_csv(file_1)
    f1_labels = file_1_pd['label'].tolist()
    f1_labels_lower = [x.lower() for x in f1_labels]
    f1_labels = replace_label_in_diet_plan(f1_labels_lower)
    f1_values = file_1_pd['quantity'].tolist()
    return list(zip(f1_labels, f1_values))


def macro_values_diet_plan(diet_data):
    for element, quantity in diet_data:
        # print(element, quantity)
        if element.strip() == 'fat':
            fat_quantity = quantity
        if element.strip() == 'protein':
            protein_quantity = quantity
        if element.strip() == 'carbohydrates':
            carb_quantity = quantity
        if element.strip() == 'calories':
            calorie_quantity = quantity
    macro_diet_db = [fat_quantity, carb_quantity, protein_quantity, calorie_quantity]
    # print(macro_diet_db)
    return macro_diet_db


def replace_label_in_diet_plan(diet_labels):
    replace_with = [
        ['niacin', 'niacin b3'],
        ['riboflavin', 'riboflavin b2'],
        ['thiamin', 'thiamin b1'],
        ['vitamin b-6', 'vitamin b6'],
        ['proteins', 'protein'],
        ['fats', 'fat']
    ]
    # for label in diet_labels:
    for i in range(0, len(diet_labels)):
        for item in enumerate(replace_with):
            if item[1][0] == diet_labels[i]:
                diet_labels[i] = item[1][1]

    return diet_labels


def populate_dict(nutrional_values, bin_values):
    new_dict = {key: [] for key in bin_values}
    for i, fat_value in enumerate(nutrional_values):
        for itr in range(0, len(bin_values) - 1):
            min_value = bin_values[itr]
            max_value = bin_values[itr + 1]
            # print('min', min_value, 'max', max_value)
            # print()
            if min_value <= fat_value < max_value:
                # print('min value', min_value, 'max value', max_value, 'fat value', fat_value)
                new_dict.setdefault(min_value).append(i)

    return new_dict


def add_id_to_bins(recipe_db_minimized_csv, fat_bins, carb_bins, protein_bins, combo_arr_fat, combo_arr_carb,
                   comb_arr_protein, diet_day, macro_diet_db):
    """ FAT DICTIONARY CREATION """
    fat_dict = populate_dict(recipe_db_minimized_csv['fat'], fat_bins)

    empty_key_fat = []
    for key, value in fat_dict.items():
        if len(value) == 0:
            empty_key_fat.append(key)

    new_fat_combos = []
    for fat_combo in combo_arr_fat:
        count = 0
        for element in empty_key_fat:

            if element in fat_combo:
                count = count + 1
        if count == 0:
            new_fat_combos.append(fat_combo)
    # print('fat combo', len(combo_arr_fat), 'new fat combo', len(new_fat_combos))

    """CARB DICTIONARY CREATION """
    carb_dict = populate_dict(recipe_db_minimized_csv['carbohydrates'], carb_bins)

    empty_key_carb = []
    for key, value in carb_dict.items():
        if len(value) == 0:
            empty_key_carb.append(key)

    new_carb_combos = []
    for carb_combo in combo_arr_carb:
        count = 0
        for element in empty_key_carb:

            if element in carb_combo:
                count = count + 1
        if count == 0:
            new_carb_combos.append(carb_combo)
    # print('carb combo', len(combo_arr_carb), 'new carb combo', len(new_carb_combos))

    """PROTEIN DICTIONARY CREATION """
    protein_dict = populate_dict(recipe_db_minimized_csv['protiens'], protein_bins)

    empty_key_protein = []
    for key, value in protein_dict.items():
        if len(value) == 0:
            empty_key_protein.append(key)

    new_protein_combos = []
    for protein_combo in comb_arr_protein:
        count = 0
        for element in empty_key_protein:

            if element in protein_combo:
                count = count + 1
        if count == 0:
            new_protein_combos.append(protein_combo)
    # print('protein combo', len(comb_arr_protein), 'new protein combo', len(new_protein_combos))

    all_combs = test_intersect_carb_and_fat_only(new_fat_combos, new_carb_combos, fat_dict, carb_dict)

    recipe_ids = []
    all_three_combinations = intersect_combinations_and_protein(all_combs, protein_dict, new_protein_combos)
    # print('+++++++')
    for item in all_three_combinations:
        # print('~~~~')
        index = []
        # print(item)
        recipe_ids = []
        for value in item:
            index.append(list(value)[0])
        # print('indexes are', index)
        i, j, k = index[0], index[1], index[2]
        recipe_ids.append([i, j, k])
        # print('recipe ids are', recipe_ids)
        # print()
        # print('all fat', recipe_db_minimized_csv.iloc[i]['fat'], recipe_db_minimized_csv.iloc[j]['fat'],
        #       recipe_db_minimized_csv.iloc[k]['fat'])
        # print('cumulative fat with id-', i, '-', j, '-', k, '-',
        #       recipe_db_minimized_csv.iloc[i]['fat'] + recipe_db_minimized_csv.iloc[j]['fat'] +
        #       recipe_db_minimized_csv.iloc[k]['fat'])
        # print()
        # print('all carb', recipe_db_minimized_csv.iloc[i]['carbohydrates'],
        #       recipe_db_minimized_csv.iloc[j]['carbohydrates'],
        #       recipe_db_minimized_csv.iloc[k]['carbohydrates'])
        # print('cumulative carb with id-', i, '-', j, '-', k, '-', recipe_db_minimized_csv.iloc[i]['carbohydrates'] +
        #       recipe_db_minimized_csv.iloc[j]['carbohydrates'] +
        #       recipe_db_minimized_csv.iloc[k]['carbohydrates'])
        # print()
        # print('all protein', recipe_db_minimized_csv.iloc[i]['protiens'], recipe_db_minimized_csv.iloc[j]['protiens'],
        #       recipe_db_minimized_csv.iloc[k]['protiens'])
        # print('cumulative protein with id-', i, '-', j, '-', k, '-',
        #       recipe_db_minimized_csv.iloc[i]['protiens'] + recipe_db_minimized_csv.iloc[j]['protiens'] +
        #       recipe_db_minimized_csv.iloc[k]['protiens'])
        print_recipe_names(recipe_ids, recipe_db_minimized_csv, diet_day, macro_diet_db)
    # print(recipe_ids)
    # suggested_recipes = print_recipe_names(recipe_ids, recipe_db_minimized_csv)
    return recipe_ids


def print_recipe_names(recipe_ids, recipe_db_minimized_csv, diet_day, macro_diet_db):
    global global_recommneded_id
    recipe_file = read_csv('recipes.csv')
    # print('inside print recipe names')
    recipe_names = []
    # print('recipe ids are', recipe_ids)
    recipe_record = []
    count = 1
    for ids in recipe_ids:
        print('ids are', ids)
        # ids = ids - 1
        all_fat = 0
        all_carb = 0
        all_protein = 0
        all_energy = 0

        for i in range(0, len(ids)):
            print(ids[i])
            print('i is', i)
            print(diet_day, count, global_recommneded_id,recipe_file['name'][ids[i]], recipe_db_minimized_csv.iloc[ids[i]]['fat'],
                  recipe_db_minimized_csv.iloc[ids[i]]['carbohydrates'],
                  recipe_db_minimized_csv.iloc[ids[i]]['protiens'])



            all_fat = all_fat + recipe_db_minimized_csv.iloc[ids[i]]['fat']
            all_carb = all_carb + recipe_db_minimized_csv.iloc[ids[i]]['carbohydrates']
            all_protein = all_protein + recipe_db_minimized_csv.iloc[ids[i]]['protiens']
            all_energy = all_energy + recipe_db_minimized_csv.iloc[ids[i]]['calories']


            # recipe_record.append([diet_day, recipe_file['name'][ids[i]], recipe_db_minimized_csv.iloc[ids[i]]['fat'],
            #       recipe_db_minimized_csv.iloc[ids[i]]['carbohydrates'], recipe_db_minimized_csv.iloc[ids[i]]['protiens']])




            from csv import writer
            with open('recipe_suggestions_kash_v2.csv', 'a', newline='', encoding="utf-8") as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(
                    [diet_day, count, global_recommneded_id, recipe_file['name'][ids[i]], recipe_db_minimized_csv.iloc[ids[i]]['fat'],
                     recipe_db_minimized_csv.iloc[ids[i]]['carbohydrates'],
                     recipe_db_minimized_csv.iloc[ids[i]]['protiens'], recipe_db_minimized_csv.iloc[ids[i]]['calories']])

                f_object.close()
            count = count + 1
            print('end of inside for loop')


        fat_diff, carb_diff, protein_diff, total_diff, calorie_diff = 0, 0, 0 , 0 , 0
        # write new csv here for difference
        # macro_diet_db = [fat_quantity, carb_quantity, protein_quantity, calorie_quantity]
        fat_diff = macro_diet_db[0] - all_fat
        carb_diff = macro_diet_db[1] - all_carb
        protein_diff = macro_diet_db[2] - all_protein
        calorie_diff = macro_diet_db[3] - all_energy
        total_diff = fat_diff + carb_diff + protein_diff
        # recipe_recommendations_kash_analysis
        # Diet Day	Global id	fat diff	carb diff	protein diff	total diff	calorie diff
        with open('recipe_recommendations_kash_analysis.csv', 'a', newline='', encoding="utf-8") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(
                [diet_day, global_recommneded_id, fat_diff, carb_diff, protein_diff, total_diff,calorie_diff])

            f_object.close()



        global_recommneded_id = global_recommneded_id + 1






            # print(recipe_db_minimized_csv.iloc[i]['carbohydrates'])
            # print(recipe_db_minimized_csv.iloc[i]['protiens'])
            # print(recipe_db_minimized_csv.iloc[i]['fat'])
            # recipe_names.append(recipe_file['name'][ids[i]])
            # print(recipe_file['nutritionalInfo'][ids[i]  ])
        # print()
    # print('****************')
    # if len(recipe_names) > 0 :
    #     for names in recipe_names:
    #         print(names)
    # else:
    #     print('None Found')
    # print(recipe_record)

    return recipe_record


def intersect_combinations_and_protein(all_combs_fat_carb, proteins_dict, new_protein_combos):
    # print('************************************************************')
    # print('inside intersect_combinations_and_protein')
    # print(proteins_dict)
    compare_combinations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    # print()
    all_combs = []
    #
    for protein_dict_key_id in new_protein_combos:

        for comb_k in all_combs_fat_carb:
            # print('comb_k', comb_k)
            #         # 0,0- 1,1 - 0,2
            for comb in compare_combinations:
                set_1 = set(proteins_dict[protein_dict_key_id[comb[0]]]) & comb_k[0]
                set_2 = set(proteins_dict[protein_dict_key_id[comb[1]]]) & comb_k[1]
                set_3 = set(proteins_dict[protein_dict_key_id[comb[2]]]) & comb_k[2]
                # print('with combination of', comb[0], comb[1], comb[2])
                # print('set-1', set_1, 'set-2', set_2, 'set 3', set_3)
                bool_array_1 = [len(set_1) > 0, len(set_2) > 0, len(set_3) > 0]
                if bool_array_1 == [True, True, True]:
                    all_combs.append([set_1, set_2, set_3])

    return all_combs


def test_intersect_carb_and_fat_only(fat_combo, carb_combo, fat_dict, carb_dict):
    compare_combinations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    # print()
    all_combs = []

    for fat_dict_key_id in fat_combo:
        # print('processing')
        for carb_dict_key_id in carb_combo:
            # 0,0- 1,1 - 0,2
            # print('processing')
            for comb in compare_combinations:
                # print(comb[0], comb[1], comb[2])
                set_1 = set(fat_dict[fat_dict_key_id[comb[0]]]) & set(carb_dict[carb_dict_key_id[0]])
                set_2 = set(fat_dict[fat_dict_key_id[comb[1]]]) & set(carb_dict[carb_dict_key_id[1]])
                set_3 = set(fat_dict[fat_dict_key_id[comb[2]]]) & set(carb_dict[carb_dict_key_id[2]])
                bool_array_1 = [len(set_1) > 1, len(set_2) > 1, len(set_3) > 1]
                if bool_array_1 == [True, True, True]:
                    all_combs.append([set_1, set_2, set_3])
    return all_combs


def intersect_carb_and_fat_only(fat_combo, carb_combo, fat_dict, carb_dict):
    all_combs = []
    for fat_dict_id in fat_combo:
        # flag_1, flag_2 , flag_3 = False, False, False
        for carb_dict_id in carb_combo:
            # compare 0 with 0, then compare 1 with 1, then compare 2 with 2
            # print('** START **')
            set_1 = set(fat_dict[fat_dict_id[0]]) & set(carb_dict[carb_dict_id[0]])
            # print(set_1)

            if len(set_1):

                set_2 = set(fat_dict[fat_dict_id[1]]) & set(carb_dict[carb_dict_id[1]])
                if len(set_2):
                    set_3 = set(fat_dict[fat_dict_id[0]]) & set(carb_dict[carb_dict_id[2]])
                    if len(set_3):
                        return

    # print('all combinations are')
    # print(all_combs)
    return all_combs


def find_recipes_change_bin_sizes(macro_diet_db, recipe_db_minimized_csv, diet_day):
    import math
    bin_sizes = [[25, 100, 2], [10, 100, 4], [20, 80, 2], [10, 90, 4], [25, 95, 3]]
    # bin_sizes = [[25, 100, 2], [10, 100, 4], [20, 80, 2]]
    fat_max = math.ceil(int(macro_diet_db[0]) / 100) * 100
    carb_max = math.ceil(int(macro_diet_db[1]) / 100) * 100
    protein_max = math.ceil(int(macro_diet_db[2]) / 50) * 50
    # print(macro_diet_db)
    # print(fat_max, carb_max, protein_max)
    for bin_size in bin_sizes:
        # print('LOOP STARTED')
        # print('checking with bin size', bin_size)
        fat_bins = np.arange(0, fat_max, bin_size[0])
        carb_bins = np.arange(0, carb_max, bin_size[1])
        protein_bins = np.arange(0, protein_max, bin_size[2])
        find_recipe_id(fat_bins, carb_bins, protein_bins, macro_diet_db, recipe_db_minimized_csv, diet_day)

        # print('LOOP FINISHED')
        # print()


def find_recipe_id(fat_bins, carb_bins, protein_bins, macro_diet_db, recipe_db_minimized_csv, diet_day):
    combo_arr_fat = [i for i in combinations(fat_bins, 3) if
                     sum(i) > macro_diet_db[0] - 15 and sum(i) < macro_diet_db[0] + 15]

    combo_arr_carb = [i for i in combinations(carb_bins, 3) if
                      sum(i) > macro_diet_db[1] - 100 and sum(i) < macro_diet_db[1] + 100]

    combo_arr_protein = [i for i in combinations(protein_bins, 3) if
                         sum(i) > macro_diet_db[2] - 7 and sum(i) < macro_diet_db[2] + 7]

    add_id_to_bins(recipe_db_minimized_csv, fat_bins, carb_bins, protein_bins, combo_arr_fat, combo_arr_carb,
                   combo_arr_protein, diet_day, macro_diet_db)

    return


def test_seven_days():
    recipe_db_minimized_csv = read_csv('recipe_db_minimized_v2.csv')
    for i in range(1, 8):
        print('Testing for Diet day ' + str(i))
        diet_data = diet_plan_day('diet_plan_day_' + str(i) + '.csv')
        macro_diet_db = macro_values_diet_plan(diet_data)
        print('macros for this diet day', macro_diet_db)
        find_recipes_change_bin_sizes(macro_diet_db, recipe_db_minimized_csv, i)
        print('END of testing')
        print()
        print()


test_seven_days()



'''NOTE : this has been run once. File is ready, no need to re-run it'''
# 1. re-arrange the recipe DB for nutritional labels and quantities
# recipe_file = read_csv('recipes.csv')
# re_arrange_recipe_db_calories_quantity(recipe_file)
# re_arrange_recipe_db(recipe_file)

# 2. Read the diet-plan day 1
##diet_data = diet_plan_day('diet_plan_day_3.csv')
##macro_diet_db = macro_values_diet_plan(diet_data)

# read the minimized csv
##recipe_db_minimized_csv = read_csv('recipe_db_minimized.csv')

# 3. use macro_diet_db and recipe_db_minimized_csv to make comparisons
# find_recipes(macro_diet_db, recipe_db_minimized_csv)
# find_recipe_id(macro_diet_db, recipe_db_minimized_csv)

# Testing
##print('TESTING FOR LOOP')
##find_recipes_change_bin_sizes(macro_diet_db, recipe_db_minimized_csv)
