import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from random import choice
from functions import calculate_initial_candle,calculate_trend_lines,plot_lines


if __name__ == "__main__":
    #Escogemos aleatoriamente el ticker de la empresa 
    #ticker = choice(["AAPL","TSLA","MSFT","GOOG","NIO","META","X","EURUSD=X"])
    ticker = "MS"
    #Descargamos los datos diarios del último año
    data = yf.download(ticker,period = "1Y",interval = "1d")
    #Resetamos el indice para que sea numerico
    data = data.reset_index()
    #Vamos a calcular el intervalo en el que queremos nuestras lineas de tendencia
    start_candle = calculate_initial_candle(data)
    #Acotamos los datos comenzando 2 previas al ultimo intervalo
    df = data.iloc[start_candle-2:].reset_index()
    #Calculamos las linas de tendencia
    trend_lines = calculate_trend_lines(df)
    #Ahora hacemos un plot de nuestras lineas de tendencia
    plot_lines(trend_lines,data.iloc[start_candle-40:],start_candle)