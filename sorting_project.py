import csv
import multiprocessing
import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000,]

RUN_COUNT = 5
MAX_VALUE = 2_147_483_647
MAX_RUNTIME_SECONDS = 20 * 60

RESULTS_FILE = "sorting_results.csv"
GRAPH_FILE = "sorting_performance.png"


def insertion_sort(values):
    """Sort values in ascending order using insertion sort."""
    for i in range(1, len(values)):
        current_value = values[i]
        j = i - 1

        while j >= 0 and values[j] > current_value:
            values[j + 1] = values[j]
            j -= 1

        values[j + 1] = current_value

    return values


def selection_sort(values):
    """Sort values in ascending order using selection sort."""
    for first in range(len(values) - 1):
        min_index = first

        for i in range(first + 1, len(values)):
            if values[i] < values[min_index]:
                min_index = i

        values[first], values[min_index] = (values[min_index],values[first])

    return values


def bubble_sort(values):
    """Sort values in ascending order using bubble sort."""
    for last in range(len(values) - 1, 0, -1):
        swapped = False

        for i in range(last):
            if values[i] > values[i + 1]:
                values[i], values[i + 1] = (values[i + 1], values[i])

                swapped = True

        # Stop early if the list is already sorted.
        if not swapped:
            break

    return values


def merge_sort(values):
    """Sort values in ascending order using merge sort."""
    if len(values) <= 1:
        return values

    middle = len(values) // 2

    left = values[:middle]
    right = values[middle:]

    merge_sort(left)
    merge_sort(right)

    i = 0
    j = 0
    k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            values[k] = left[i]
            i += 1
        else:
            values[k] = right[j]
            j += 1

        k += 1

    while i < len(left):
        values[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        values[k] = right[j]
        j += 1
        k += 1

    return values


def quick_sort(values):
    """Sort values in ascending order using quick sort."""
    quick_sort_recursive(values, 0, len(values) - 1)

    return values


def quick_sort_recursive(values, low, high):
    """Recursively sort values from low through high."""
    if low < high:
        pivot_index = partition(values, low, high)
        quick_sort_recursive(values, low, pivot_index - 1)
        quick_sort_recursive(values, pivot_index + 1, high)


def partition(values, low, high):
    """Partition values around the element at index high."""
    pivot = values[high]
    smaller_index = low - 1

    for current_index in range(low, high):
        if values[current_index] <= pivot:
            smaller_index += 1

            values[smaller_index], values[current_index] = (values[current_index], values[smaller_index])

    values[smaller_index + 1], values[high] = (values[high], values[smaller_index + 1])
    return smaller_index + 1


def test_sorting_algorithms():
    """Test all sorting algorithms using two random lists."""
    algorithms = [("Insertion Sort", insertion_sort),("Selection Sort", selection_sort), ("Bubble Sort", bubble_sort), ("Merge Sort", merge_sort), ("Quick Sort", quick_sort)]

    for test_number in range(1, 3):
        original = [random.randint(0, 100) for _ in range(10)]

        print(f"\nTest {test_number}", flush=True)

        print("Original list:", original, flush=True)

        for algorithm_name, algorithm in algorithms:
            test_values = original.copy()
            algorithm(test_values)

            print(f"{algorithm_name}: {test_values}",flush=True)


def sorting_worker(sort_function, values, connection):
    """Run one sorting trial and send its execution time."""
    try:
        start_time = time.perf_counter()
        sort_function(values)
        end_time = time.perf_counter()
        sorting_time = end_time - start_time
        connection.send(("completed", sorting_time))

    except Exception as error:
        connection.send(("error", str(error)))

    finally:
        connection.close()


def run_single_trial(sort_function, size, run_number):
    """Generate data and execute one timed sorting trial."""
    print(f"Run {run_number}: generating random data...", flush=True)

    generation_start = time.perf_counter()
    values = [random.randint(0, MAX_VALUE) for _ in range(size)]
    generation_end = time.perf_counter()
    generation_time = (generation_end - generation_start)

    print(f"Run {run_number}: data generation took " f"{generation_time:.6f} seconds", flush=True)
    print(f"Run {run_number}: sorting...", flush=True)

    # The Indiana University BigRed200 is Linux, so we are using fork.
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = (context.Pipe(duplex=False))

    process = context.Process(target=sorting_worker, args=(sort_function, values, child_connection))
    wall_start = time.perf_counter()

    process.start()
    child_connection.close()

    # Waiting at most 20 minutes for a result from the worker.
    result_available = parent_connection.poll(MAX_RUNTIME_SECONDS)

    if not result_available:
        print(f"Run {run_number}: exceeded 20 minutes.", flush=True)

        process.terminate()
        process.join(timeout=5)

        if process.is_alive():
            process.kill()
            process.join()

        parent_connection.close()

        return None

    status, result = parent_connection.recv()
    parent_connection.close()
    process.join()
    wall_end = time.perf_counter()
    wall_time = wall_end - wall_start

    if status == "error":
        print(f"Run {run_number}: sorting failed: {result}", flush=True)
        return None

    sorting_time = result

    print(f"Run {run_number}: sorting time = " f"{sorting_time:.6f} seconds", flush=True)
    print(f"Run {run_number}: worker wall time = " f"{wall_time:.6f} seconds", flush=True)

    return sorting_time


def measure_performance(sort_function, size):
    """Run five trials and average all except the slowest."""
    execution_times = []

    for run_number in range(1, RUN_COUNT + 1):
        sorting_time = run_single_trial(sort_function, size, run_number)

        if sorting_time is None:
            return None

        execution_times.append(sorting_time)

    slowest_time = max(execution_times)
    execution_times.remove(slowest_time)

    average_time = (sum(execution_times) / len(execution_times))

    return average_time


def save_results(results, statuses):
    """Save collected benchmark results to a CSV file."""
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)

        writer.writerow(["Algorithm", "Dataset Size", "Status", "Average Time (Seconds)"])

        for algorithm_name in results:
            for size in DATASET_SIZES:
                average_time = results[algorithm_name].get(size)

                status = statuses[algorithm_name].get(size,"Not tested")

                if average_time is None:
                    time_value = ""
                else:
                    time_value = (f"{average_time:.6f}")

                writer.writerow([algorithm_name, size, status, time_value])


def run_performance_tests():
    """Run all required performance tests."""
    algorithms = [("Insertion Sort", insertion_sort), ("Selection Sort", selection_sort), ("Bubble Sort", bubble_sort), ("Merge Sort", merge_sort), ("Quick Sort", quick_sort)]

    results = {}
    statuses = {}

    for algorithm_name, algorithm in algorithms:
        results[algorithm_name] = {}
        statuses[algorithm_name] = {}

        print(f"\n{algorithm_name}", flush=True)

        for size in DATASET_SIZES:
            print(f"\nDataset size: {size:,}", flush=True)

            average_time = measure_performance(algorithm, size)
            results[algorithm_name][size] = average_time

            if average_time is None:
                statuses[algorithm_name][size] = ("Exceeded 20 minutes")

                print(f"{algorithm_name} at " f"{size:,}: > 20 m", flush=True,)

                # Skipping larger datasets for this algorithm.
                for larger_size in DATASET_SIZES:
                    if larger_size > size:
                        results[algorithm_name][larger_size] = None
                        statuses[algorithm_name][larger_size] = ("Skipped after cutoff")

                save_results(results, statuses)

                break

            statuses[algorithm_name][size] = "Completed"
            print(f"Average time: " f"{average_time:.6f} seconds", flush=True)
            save_results(results, statuses)

    return results, statuses


def plot_results(results):
    """Create and save a graph of average sorting runtimes."""
    plt.figure(figsize=(10, 6))

    for algorithm_name, timings in results.items():
        dataset_sizes = []
        average_times = []

        for size in DATASET_SIZES:
            average_time = timings.get(size)

            if average_time is not None:
                dataset_sizes.append(size)
                average_times.append(average_time)

        if dataset_sizes:
            plt.plot(dataset_sizes, average_times, marker="o", label=algorithm_name)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Dataset Size")
    plt.ylabel("Average Runtime (Seconds)")

    plt.title("Sorting Algorithm Performance")
    plt.legend()
    plt.grid(True,which="both")
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"\nGraph saved as {GRAPH_FILE}", flush=True)


def print_final_results(results, statuses):
    """Display completed, timed-out, and skipped results."""
    print("\nFinal Results", flush=True)

    for algorithm_name in results:
        print(f"\n{algorithm_name}", flush=True)

        for size in DATASET_SIZES:
            status = statuses[algorithm_name].get(size, "Not tested")
            average_time = results[algorithm_name].get(size)

            if status == "Completed":
                print(f"{size:,}: " f"{average_time:.6f} seconds", flush=True)

            elif status == "Exceeded 20 minutes":
                print(f"{size:,}: > 20 minutes", flush=True)

            elif status == "Skipped after cutoff":
                print(f"{size:,}: " f"not tested after cutoff", flush=True)

            else:
                print(f"{size:,}: {status}", flush=True)


def main():
    """Run correctness tests, benchmarks, and graph generation."""

    test_sorting_algorithms()
    print("\nPerformance Testing", flush=True)
    results, statuses = (run_performance_tests())
    print_final_results(results, statuses)
    plot_results(results)


if __name__ == "__main__":
    main()

    