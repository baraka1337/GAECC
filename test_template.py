from main import *
import os.path


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
        "num_parents_mating": int(600*0.2),
        "offspring_size": int(600*0.8),
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


def test_template(param):
    name = "_".join(f"{a}_{b}" for a, b in param.items())
    filename = name + ".pickle"
    if not os.path.exists(filename):
        ga = GA(**param)
        ga.run()
        ga.dump_to_file(filename)
    else:
        ga = GA.load_from_file(filename)
        ga.end()

    plot_best(ga, name)
    plot_all_generations(ga, name)


def plot_best(ga, name):
    best_of_the_bests = np.argmax(ga.bests_nlog)
    plt.figure()
    plt.plot(ga.bests_nlog)
    plt.plot(best_of_the_bests,
             ga.bests_nlog[best_of_the_bests], "o", color="red", label=f"Best: {ga.bests_nlog[best_of_the_bests]:.2f}")
    plt.xlabel("Generation")
    plt.ylabel("-Log(ber)")
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(f"{name}_best.png")


def plot_all_generations(ga, name):
    plt.figure()
    for i, nlog in enumerate(ga.all_nlog):
        if i == 0 or (i+1) % 5 == 0:
            plt.plot(sorted(nlog), label=f"gen {i+1}")
    plt.grid()
    plt.xlabel("Population")
    plt.ylabel("-Log(ber)")
    plt.legend()
    plt.savefig(f"{name}_all.png")


if __name__ == "__main__":
    print("Test 9:")
    test_9()
    print("Test 10:")
    test_10()
    print("Test 11:")
    test_11()
    print("Test 12:")
    test_12()
    print("Test 13:")
    test_13()