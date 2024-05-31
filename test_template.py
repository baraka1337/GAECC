from main import *
from code import snr_to_EbN0  # type: ignore
import os.path
import glob
import csv

from utils import plot_best, best_matrix_to_image


def test_0():
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 100,
        "sample_size": 100,
        "p_mutation": 0.01,
        "num_generations": 2,
        "ebn0": 4,
        "num_parents_mating": 20,
        "offspring_size": 80,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_pickle():
    # k_16_n_31_num_initial_population_1000_sample_size_10000_
    # p_mutation_0.004166666666666667_num_generations_80_ebn0_4_
    # num_parents_mating_100_offspring_size_700_delta_1e-20_
    # gamma_7_bp_iter_5.pickle
    param = {
        "k": 16,
        "n": 31,
        "num_initial_population": 1000,
        "sample_size": 10000,
        "p_mutation": 1 / (16 * (31 - 16)),
        "num_generations": 80,
        "ebn0": 4,
        "num_parents_mating": 100,
        "offspring_size": 700,
        "delta": 1e-20,
        "gamma": 7,
        "bp_iter": 5,
    }
    test_template(param)


def test_1():
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_2():
    # more mutation
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.05,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_2_5():
    # less mutation
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.005,
        "num_generations": 100,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_2_5_5():
    # less mutation
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.001,
        "num_generations": 100,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_3():
    # more parents_mating
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 400,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_3():
    # more offspring
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 900,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_4():
    # more gamma
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 4,
    }
    test_template(param)


def test_5():
    # less gamma
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_6():
    # less parents_mating
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 400,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_7():
    # less parents_mating
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 100,
        "ebn0": 4,
        "num_parents_mating": 400,
        "offspring_size": 900,
        "delta": 1e-20,
        "gamma": 5,
    }
    test_template(param)


def test_8():
    # less parents_mating
    param = {
        "k": 4,
        "n": 7,
        "num_initial_population": 100,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 40,
        "offspring_size": 90,
        "delta": 1e-20,
        "gamma": 5,
    }
    test_template(param)


def test_9():
    # less initial population
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 600,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": int(600 * 0.2),
        "offspring_size": int(600 * 0.8),
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_10():
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 10,
    }
    test_template(param)


def test_11():
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 7,
    }
    test_template(param)


def test_12():
    # more offspring
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 200,
        "offspring_size": 1000,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_13():
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 50,
        "ebn0": 4,
        "num_parents_mating": 100,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 3,
    }
    test_template(param)


def test_all():
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 10000,
        "p_mutation": 0.01,
        "num_generations": 80,
        "ebn0": 4,
        "num_parents_mating": 100,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 7,
        "bp_iter": 5,
    }

    for o_size in [800]:
        for bp_iter in [5]:
            for e in [5]:
                for n, k in [(63, 45), (49, 24), (60, 52), (64, 32)]:
                    param["k"] = k
                    param["n"] = n
                    param["ebn0"] = e
                    param["p_mutation"] = 1 / (k * (n - k))
                    param["offspring_size"] = o_size
                    param["bp_iter"] = bp_iter
                    # if e == 6 or (e == 5 and n == 64):
                    #     param["sample_size"] = param["sample_size"]
                    print(
                        f"Running test for k={param['k']}, n={param['n']}, ebn0={param['ebn0']}, bp_iter={param['bp_iter']}"
                    )
                    test_template(param, folder="results")

    n_k = [(32, 11), (31, 16), (63, 45), (49, 24), (60, 52), (64, 32)]
    ebn0 = [5, 4]
    for o_size in [800, 700]:
        for bp_iter in [50, 5]:
            for e in ebn0:
                for n, k in n_k:
                    param["k"] = k
                    param["n"] = n
                    param["ebn0"] = e
                    param["p_mutation"] = 1 / (k * (n - k))
                    param["offspring_size"] = o_size
                    param["bp_iter"] = bp_iter
                    # if e == 6 or (e == 5 and n == 64):
                    #     param["sample_size"] = param["sample_size"]
                    print(
                        f"Running test for k={param['k']}, n={param['n']}, ebn0={param['ebn0']}, bp_iter={param['bp_iter']}"
                    )
                    test_template(param, folder="results")

    for o_size in [700]:
        for bp_iter in [5]:
            for e in [5]:
                for n, k in [(63, 45), (49, 24), (60, 52), (64, 32)]:
                    param["k"] = k
                    param["n"] = n
                    param["ebn0"] = e
                    param["p_mutation"] = 1 / (k * (n - k))
                    param["offspring_size"] = o_size
                    param["bp_iter"] = bp_iter
                    # if e == 6 or (e == 5 and n == 64):
                    #     param["sample_size"] = param["sample_size"]
                    print(
                        f"Running test for k={param['k']}, n={param['n']}, ebn0={param['ebn0']}, bp_iter={param['bp_iter']}"
                    )
                    test_template(param, folder="results")


def test_template(param, folder="."):
    name = "_".join(f"{a}_{b}" for a, b in param.items())
    filename = os.path.join(folder, name + ".pickle")
    if not os.path.exists(filename):
        ga = GA(**param)
        ga.run()
        ga.dump_to_file(filename)
    else:
        ga = GA.load_from_file(filename)

    filename_figure = os.path.join(folder, name)
    plot_best(ga, filename_figure)
    best_matrix_to_image(ga, filename_figure)
    # plot_all_generations(ga, filename_figure)


def summarize_reuslts_in_excel(
    dir_path="./", output_file_path="./best_results_summary.csv"
):
    param = {
        "k": 24,
        "n": 49,
        "num_initial_population": 1000,
        "sample_size": 1000,
        "p_mutation": 0.01,
        "num_generations": 100,
        "ebn0": 4,
        "num_parents_mating": 100,
        "offspring_size": 800,
        "delta": 1e-20,
        "gamma": 7,
    }
    pattern = os.path.join(dir_path, "*.pickle")
    data = []
    for filepath in glob.iglob(pattern):
        ga = GA.load_from_file(filepath)
        ga_data = {
            key: value
            for key, value in ga.__dict__.items()
            if key in param.keys() and key != "p_mutation"
        }
        ga_data["avg_err_num"] = ga.p_mutation * ga.k * (ga.n - ga.k)
        best_of_the_bests = np.argmax(ga.bests_nlog)
        ga_data["ebn0"] = snr_to_EbN0(ga.snr, ga.k / ga.n)
        ga_data["best_of_the_bests_generation"] = best_of_the_bests
        ga_data["best_of_the_bests -Log(BER)"] = ga.bests_nlog[best_of_the_bests]
        ga_data["bp_iter"] = ga.__dict__.get("bp_iter", 5)
        if ga.__dict__.get("bp_iter", False):
            print("HEHR")
            ga_data["has_best_solution"] = "V"
        else:
            ga_data["has_best_solution"] = "X"
        data.append(ga_data)

    # Get the header from the keys of the first dictionary
    header = data[0].keys()

    # Write the data to a CSV file
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
    try:
        import pandas as pd
    except:
        os.system("pip install pandas")
        import time

        time.sleep(10)
        import pandas as pd
    df = pd.read_csv(output_file_path)
    df["Best?"] = "X"
    print(df.bp_iter.drop_duplicates())
    print(
        [
            (n, k)
            for n, k in zip(
                df.get(["k", "n"]).drop_duplicates().n,
                df.get(["k", "n"]).drop_duplicates().k,
            )
        ]
    )
    for n, k in zip(
        df.get(["k", "n"]).drop_duplicates().n, df.get(["k", "n"]).drop_duplicates().k
    ):
        for bp_iter in df.bp_iter.drop_duplicates():
            if bp_iter != 0:
                for e in df.ebn0.drop_duplicates():
                    cond1 = df["k"] == k
                    cond2 = df["n"] == n
                    cond3 = df["ebn0"] == e
                    cond4 = df["bp_iter"] == bp_iter
                    cond = [
                        cond1[i] and cond2[i] and cond3[i] and cond4[i]
                        for i in range(len(cond1))
                    ]
                    if not df[cond]["best_of_the_bests -Log(BER)"].empty:
                        max_ind = df[cond].index[
                            np.argmax(df[cond]["best_of_the_bests -Log(BER)"])
                        ]
                        df.iloc[max_ind, -1] = "V"
                        # print(df.iloc[max_ind])
    df.to_csv(output_file_path)
    print(f"Data has been written to {output_file_path}")


if __name__ == "__main__":
    test_pickle()
