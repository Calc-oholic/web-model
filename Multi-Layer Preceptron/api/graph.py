# graph.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta

from api.model import unrelativize_data_array


def plot_predictions(plotted_context, plotted_solution, predictions, context_window):
    plotted_predictions = predictions.reshape(-1, 1)
    plt.plot(plotted_context, "blue", label='Context')

    # Actual
    x_solution_values = range(context_window - 1, context_window + len(plotted_solution) - 1)
    plt.plot(x_solution_values, plotted_solution, "green", label='Actual Solution')

    # Predictions
    x_prediction_values = range(context_window-1, context_window + len(plotted_predictions)-1)
    plt.plot(x_prediction_values, plotted_predictions, "red", label='Predictions')

    plt.xlim(len(unrelativize_data_array(plotted_context)) - len(unrelativize_data_array(plotted_predictions)) * 1.1,
             len(unrelativize_data_array(plotted_context)) + len(unrelativize_data_array(plotted_predictions)) * 1.1)

    # might not work lmaooo
    print(plt.grid())
    plt.legend()
    # plt.show()

    output_path = os.path.join("api", "static", "predictions_plot.png")
    plt.savefig(output_path)
    plt.close()

def plot_normalized(plotted_context, plotted_solution, predictions, context_window):
    plotted_predictions = predictions.reshape(-1, 1)

    plt.plot(unrelativize_data_array(plotted_context) - unrelativize_data_array(plotted_context)[-1], "blue", label="Context")

    x_values = range(context_window - 1, context_window + len(plotted_solution) - 1)
    y_values = unrelativize_data_array(plotted_solution) * (unrelativize_data_array(plotted_context)[-1] + 1)
    plt.plot(x_values, y_values, "green", label="Actual")

    x_values = range(context_window - 1, context_window + len(plotted_predictions) - 1)
    y_values = unrelativize_data_array(plotted_predictions) * (unrelativize_data_array(plotted_context)[-1] + 1)
    plt.plot(x_values, y_values, "red", label="Predictions")

    xlim = len(unrelativize_data_array(plotted_context)) - len(unrelativize_data_array(plotted_predictions)) * 1.1, len(
        unrelativize_data_array(plotted_context)) + len(unrelativize_data_array(plotted_predictions)) * 1.1
    plt.xlim(xlim)

    allValuesInPlot = np.append(unrelativize_data_array(plotted_predictions), unrelativize_data_array(plotted_solution))
    # print("Values1" + allValuesInPlot)
    allValuesInPlot = np.append(allValuesInPlot,
                                (unrelativize_data_array(plotted_context) - unrelativize_data_array(plotted_context)[-1])[
                                int(xlim[0]):int(xlim[1])])
    # print("Values2: " + allValuesInPlot)

    print(plt.grid())
    plt.legend()
    # plt.show()

    output_path = os.path.join("api", "static", "normalized_predictions_plot.png")
    plt.savefig(output_path)
    plt.close()


def plot_stock_prices(plotted_context, plotted_solution, predictions, context_window, solution_window, current_date):
    unrelativized_context = unrelativize_data_array(plotted_context)
    unrelativized_solution = unrelativize_data_array(plotted_solution)
    unrelativized_predictions = unrelativize_data_array(predictions)

    last_context_value = unrelativized_context[-1]

    # Calculate actual prices
    actual_prices = unrelativized_solution + last_context_value
    predicted_prices = unrelativized_predictions + last_context_value

    # Calculate percentage changes
    actual_percentage_changes = [(price - last_context_value) / last_context_value for price in actual_prices]
    actual_percentage_changes = np.array(actual_percentage_changes)
    predicted_percentage_changes = [(price - last_context_value) / last_context_value for price in predicted_prices]
    predicted_percentage_changes = np.array(predicted_percentage_changes)

    actual_dates = [current_date - timedelta(days=(solution_window - 1 - i)) for i in range(solution_window)]
    prediction_dates = [current_date - timedelta(days=(solution_window - 1 - i)) for i in range(solution_window)]

    # Combine dates and percentages for plotting
    x_values_actual = actual_dates
    x_values_predictions = prediction_dates

    plt.figure(figsize=(12, 6))

    # Plot actual percentage changes
    plt.plot(x_values_actual, actual_percentage_changes, "green", label="Actual % Change")

    # Plot predicted percentage changes
    plt.plot(x_values_predictions, predicted_percentage_changes, "red", label="Predicted % Change")

    # Rotate date labels for better readability
    plt.gcf().autofmt_xdate()

    plt.grid()
    plt.legend()

    # Show the plot
    plt.show()


def get_error(plotted_solution, predictions):
    plotted_predictions = predictions.reshape(-1, 1)

    avgPercentErrors = []
    avgAbsoluteErrors = []

    for i in range(len(plotted_predictions)):
        percentError = abs((plotted_predictions[i] - plotted_solution[i]) / plotted_solution[i])  # 1% = 0.01
        avgPercentErrors.append(percentError)

        absoluteError = abs(plotted_predictions[i] - plotted_solution[i])
        avgAbsoluteErrors.append(absoluteError)

    avgPercentErrors = pd.DataFrame(avgPercentErrors)
    avgPercentErrors = avgPercentErrors.replace([np.inf, -np.inf], np.nan).dropna()

    avgPercentError = sum(avgPercentErrors[0]) / len(avgPercentErrors)
    avgAbsoluteError = (sum(avgAbsoluteErrors) / len(avgAbsoluteErrors))[0]

    return avgPercentError, avgAbsoluteError