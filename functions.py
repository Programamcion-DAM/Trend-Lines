import numpy as np
import plotly.graph_objects as go

def calculate_initial_candle(df):
    #Establecemos la ventana para calcular los máximos/mínimos   
    peaks_window = 20
    #Evaluamos las condiciones para que sean máximos o mínimos
    low_conditions = df["Low"] == df["Low"].rolling(2 * peaks_window + 1,center = True).min()
    high_conditions =  df["High"] == df["High"].rolling(2 * peaks_window + 1,center = True).max()
    #Tomamos los índices que cumplan nuestras condiciones
    indexes = list(df[low_conditions].index) + list(df[high_conditions].index)
    #Los ordenamos de menor a mayor
    indexes = sorted(indexes)
    #Ahora que tenemos la lista con todas las particiones, devolvemos la última partición
    return indexes[-2]


def calculate_trend_lines(df):
    #Primeramente vamos a calcular los máximos y minimos en 5 velas
    df['High_5_candle'] = df["High"] == df["High"].rolling(5,center = True).max()
    df['Low_5_candle'] = df["Low"] == df["Low"].rolling(5,center = True).min()
    #Transformamos todo a arrays numpy y objetvos python para mayor eficiencia
    df_indices = df.index.values
    df_high_values = df["High"].values
    df_low_values = df["Low"].values
    #Creamos el objeto que contenga a los dos
    df_line_type_values = {
        "High": df_high_values,
        "Low": df_low_values
    }
    df_line_type_5_candle_values = {
        "High": df["High_5_candle"].values,
        "Low": df["Low_5_candle"].values
    }
    
    #Creamos la primera funcion
    def find_best_line(line_type):
        #Tomamos todos los indices de nuestros puntos de interes
        interest_indices = np.where(df_line_type_5_candle_values[line_type] == 1)[0]
        #Si es vacio tomamos el indice del minimo o maximo de todo el conjunto
        if len(interest_indices) == 0:
            index = np.argmax(df_line_type_values[line_type]) if line_type == "High" else np.argmin(df_line_type_values[line_type])
            interest_indices = [index]
        #Tomamos los valores de interes
        interest_values = df_line_type_values[line_type][interest_indices]
        #Vamos a calcular cuanto es la distancia para considerar que estamos cerca de una vela
        epsilon = (df_high_values - df_low_values).mean() * 0.25
        #Por cada indice, valor calculamos la linea de tendencia que origina ahi
        final_line = None
        max_weight = 0
        for x0,y0 in zip(interest_indices,interest_values):
            line, weight = calculate_line(x0, y0, line_type, epsilon)
            #Nos quedaremos con la que mayor fuerza tenga de todas
            if weight > max_weight:
                max_weight = weight
                final_line = line
        #Devolvemos la final y mas fuerte
        return final_line, max_weight
    

    def calculate_line(x0,y0,line_type,epsilon):
        #Tomamos el indice y valor del ultimo elemento del df
        x1 = df_indices[-1]
        y1 = df_line_type_values[line_type][x1]
        #Ahora que sabemos donde inicia y termina nuestra linea, vamos a ajustarla a los datos
        line, ypoints = adjust_line(x0, y0, x1, y1, line_type)
        #Segmentmamos todos los puntos Y para solo quedarnos con los de interes
        segment_indexes = df_line_type_5_candle_values[line_type][x0:] == 1
        segment_ypoints = ypoints[segment_indexes]
        segment_graph_points = df_line_type_values[line_type][x0:][segment_indexes]
        #Comparamos para saber la fuerza que tiene la linea de tendencia
        weight = np.sum(np.abs(segment_ypoints - segment_graph_points) <= epsilon)
        #Devolvemos la linea y su fuerza 
        return line, weight

    def adjust_line(x0, y0, x1, y1, line_type):
        while True:
            #Resolvemos un sistema lineal para calcular la linea
            A = np.array([[x0, 1], [x1, 1]])
            B = np.array([y0, y1])
            line = np.linalg.solve(A, B)
            #Evaluamos la linea en los puntos posteriores a nuestro incial
            ypoints = line[0] * df_indices[x0:] + line[1]
            #Comprobamos si necesitamos reajuste
            x1_new_idx = need_readjust(x0, ypoints, line_type)
            #Si no necesitamos devolvemos la linea y los puntos y calculados
            if x1_new_idx == -1:
                return line, ypoints
            #Si no establecemos el nuevo y1 y repetimos
            else:
                x1 = x1_new_idx
                y1 = df_line_type_values[line_type][x1]
    
    def need_readjust(x0,ypoints,line_type):
        #Tomamos los puntos Y reales
        line_data = df_line_type_values[line_type][x0:]
        #Calculamos la diferencia con el pronostico
        diff = line_data - ypoints
        #Vemos si la linea choca con alguna vela posterior a nuestro x0, si es asi devolvemos su indice
        if line_type == "High" and diff.max() > 1e-6:
            max_idx = diff.argmax()
            return x0 + max_idx
        if line_type == "Low" and diff.min() < -1e-6:
            min_idx = diff.argmin()
            return x0 + min_idx
        #Si no choca, hemos encontrado una linea candidata
        return -1
    
    #Finalmente calculamos las dos lineas de tendencia para el grafico
    high_line, high_weight = find_best_line("High")
    low_line, low_weight = find_best_line("Low")
    #Las devolvemos en objetivo de python
    return {
        "High": {"line": high_line, "weight": high_weight},
        "Low": {"line": low_line, "weight": low_weight}
    }

def plot_lines(trend_lines,df,start_candle):
    high_line = trend_lines['High']['line']
    low_line = trend_lines['Low']['line']
    high_weight = trend_lines['High']['weight']
    low_weight = trend_lines['Low']['weight']

    
    # Calculamos los puntos Y
    xpoints = df.loc[start_candle:].index - 2 

    if high_line is not None:
        yhpoints = high_line[0] * range(len(xpoints)) + high_line[1]
    else:
        yhpoints = None

    if low_line is not None:
        ylpoints = low_line[0] * range(len(xpoints)) + low_line[1]
    else:
        ylpoints = None

    # Creamos el grafico con la libreria plotly
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         increasing_line_color='green',
                                         decreasing_line_color='red')])

    # Añadimos la linea superior
    if yhpoints is not None:
        fig.add_trace(go.Scatter(x=xpoints,
                                 y=yhpoints,
                                 mode="lines",
                                 line=dict(color="blue"),
                                 name=f"High Trend Line, Weight: {high_weight}"))

    # Añadimos la linea inferior
    if ylpoints is not None:
        fig.add_trace(go.Scatter(x=xpoints,
                                 y=ylpoints,
                                 mode="lines",
                                 line=dict(color="orange"),
                                 name=f"Low Trend Line, Weight: {low_weight}"))
        
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')

    # Show the plot
    fig.show()
