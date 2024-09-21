# index.py
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from datetime import date
from api.model import train_model
from api.graph import plot_predictions, get_error, plot_normalized, plot_stock_prices

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        print("Ticker: " + ticker)
        print("Date Range: " + start_date + " to " + date.today().strftime('%Y-%m-%d'))
        print("Processing...")

        # solution_window sets the amount of days to predict, and it is basing that off of the context_window days before that, so total days data is context_window+solution_window
        solution_window = 30
        # this means it predicts the prior 30 days
        context_window = 300
        # based off of the 300 days prior to that 30 days of data


        predictions, mse, plotted_context, plotted_solution = train_model(ticker, start_date, context_window, solution_window)
        print("Finished!")

        current_date = date.today()

        plot_predictions(plotted_context, plotted_solution, predictions, context_window)
        plot_normalized(plotted_context, plotted_solution, predictions, context_window)
        plot_stock_prices(plotted_context, plotted_solution, predictions, context_window, solution_window, current_date)

        average_percent_error, average_absolute_error = get_error(plotted_solution, predictions)

        print(f"Average percent error: {round(average_percent_error * 100, 2)}%")
        print(f"Average absolute error: {round(average_absolute_error * 100, 2)}%")
        print(f"Mean Squared Error: {mse}")

        return render_template("graph.html", mse=mse, ticker=ticker)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)