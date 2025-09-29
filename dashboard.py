import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

SEED = 25
torch.manual_seed(SEED)

# Configuración de la página
st.set_page_config(layout="wide", page_title="Predicción de Accidentes de Tráfico", page_icon="🚗")
st.title("🚗 Predicción de Accidentes de Tráfico por Provincia y Mes")

# ==============================================
# 1. Carga y Preprocesamiento de Datos
# ==============================================
@st.cache_data
def load_and_preprocess_data():
    # Carga de datos
    df = pd.read_parquet('accidentes_trafico_final.parquet')

    # Limpieza y tratamiento de nulos
    cols_invalid = ['TITULARIDAD_VIA', 'NUDO_INFO', 'PRIORI_NORMA', 'PRIORI_AGENTE', 'PRIORI_SEMAFORO',
                    'PRIORI_VERT_STOP', 'PRIORI_VERT_CEDA', 'PRIORI_HORIZ_STOP', 'PRIORI_HORIZ_CEDA',
                    'PRIORI_MARCAS', 'PRIORI_PEA_NO_ELEV', 'PRIORI_PEA_ELEV', 'PRIORI_MARCA_CICLOS',
                    'PRIORI_CIRCUNSTANCIAL', 'PRIORI_OTRA', 'CONDICION_NIVEL_CIRCULA', 'CONDICION_FIRME',
                    'CONDICION_ILUMINACION', 'CONDICION_METEO', 'VISIB_RESTRINGIDA_POR', 'ACERA', 'TRAZADO_PLANTA']

    for c in cols_invalid:
        df[c] = df[c].replace({998: np.nan, 999: np.nan})

    num_cols_nulos = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()].tolist()
    cat_cols_nulos = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()].tolist()

    for col in num_cols_nulos:
        df[col] = df[col].fillna(0)
    for col in cat_cols_nulos:
        df[col] = df[col].fillna('missing')

    df['fecha'] = pd.to_datetime(dict(year=df.ANYO, month=df.MES, day=1))
    df['mes'] = df['fecha'].dt.month
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    monthly_counts = df.groupby(['COD_PROVINCIA', 'fecha']).size().reset_index(name='NUM_ACCIDENTE_MES')

    categorical_cols = ['CONDICION_METEO', 'CONDICION_ILUMINACION', 'CONDICION_NIEBLA', 'CONDICION_VIENTO',
                        'VISIB_RESTRINGIDA_POR', 'PRIORI_NORMA', 'PRIORI_AGENTE', 'PRIORI_MARCAS', 'PRIORI_PEA_NO_ELEV',
                        'PRIORI_PEA_ELEV', 'PRIORI_MARCA_CICLOS', 'PRIORI_CIRCUNSTANCIAL', 'PRIORI_OTRA',
                        'CONDICION_NIVEL_CIRCULA', 'CONDICION_FIRME', 'ACERA', 'TRAZADO_PLANTA',
                        'TITULARIDAD_VIA', 'NUDO_INFO', 'PRIORI_SEMAFORO', 'PRIORI_VERT_STOP', 'PRIORI_VERT_CEDA',
                        'PRIORI_HORIZ_STOP', 'PRIORI_HORIZ_CEDA', 'TIPO_VIA', 'NUDO']

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col].fillna(-1), prefix=col.lower(), prefix_sep='_', dtype=float)
        df = pd.concat([df, dummies], axis=1)

    dummy_cols = [c for c in df.columns if any(c.startswith(col.lower() + '_') for col in categorical_cols)]
    numeric_cols = ['trafico_ligero_estimado', 'trafico_pesado_estimado']  # Asegurar que estas columnas existan

    agg_df = df.groupby(['COD_PROVINCIA', 'fecha'])[dummy_cols + numeric_cols].mean().reset_index()

    df = monthly_counts.merge(agg_df, on=['COD_PROVINCIA', 'fecha'], how='left')

    # Expandir combinaciones
    provincias = df['COD_PROVINCIA'].unique()
    fechas = pd.date_range(start=df['fecha'].min(), end=df['fecha'].max(), freq='MS')
    multi_index = pd.MultiIndex.from_product([provincias, fechas], names=['COD_PROVINCIA', 'fecha'])
    df_full = df.set_index(['COD_PROVINCIA', 'fecha']).reindex(multi_index).reset_index()
    df_full['NUM_ACCIDENTE_MES'] = df_full['NUM_ACCIDENTE_MES'].fillna(0)
    df_full.fillna(0, inplace=True)
    
    # Recalcular mes y características cíclicas sobre df_full:
    df_full['mes'] = df_full['fecha'].dt.month
    df_full['mes_sin'] = np.sin(2 * np.pi * df_full['mes'] / 12)
    df_full['mes_cos'] = np.cos(2 * np.pi * df_full['mes'] / 12)

    # Añadir características temporales
    df_full['year'] = df_full['fecha'].dt.year
    min_year = df_full['year'].min()
    max_year = df_full['year'].max()
    df_full['year_scaled'] = (df_full['year'] - min_year) / max(1, (max_year - min_year))  # Evitar división por cero
    
    # Verificar y crear características faltantes
    if 'trafico_ligero_estimado' not in df_full.columns:
        df_full['trafico_ligero_estimado'] = 0
    if 'trafico_pesado_estimado' not in df_full.columns:
        df_full['trafico_pesado_estimado'] = 0
    
    # Definir características clave
    key_features = ['mes_sin', 'mes_cos', 'year_scaled']
    
    # Añadir características de tráfico si existen
    if 'trafico_ligero_estimado' in df_full.columns:
        key_features.append('trafico_ligero_estimado')
    if 'trafico_pesado_estimado' in df_full.columns:
        key_features.append('trafico_pesado_estimado')
    
    return df_full, key_features

# Cargar datos y características
df_full, KEY_FEATURES = load_and_preprocess_data()

# ==============================================
# 2. Modelo LSTM Mejorado
# ==============================================
class ImprovedLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# ==============================================
# 3. Funciones de Entrenamiento y Pronóstico
# ==============================================
def train_province(df, prov_code, lookback, key_features, epochs=100, batch_size=32, lr=0.001):
    dfp = df[df['COD_PROVINCIA'] == prov_code].sort_values('fecha').reset_index(drop=True)
    
    # Verificar que todas las características existan
    missing_features = [feat for feat in key_features if feat not in dfp.columns]
    if missing_features:
        st.warning(f"Advertencia: Características faltantes: {missing_features}")
        # Mantener solo las características disponibles
        key_features = [feat for feat in key_features if feat in dfp.columns]
    
    # Usar solo características clave disponibles
    X = dfp[key_features].values
    y = dfp['NUM_ACCIDENTE_MES'].values
    
    if len(X) < lookback + 1:
        raise ValueError(f"No hay suficientes datos para provincia {prov_code}")
    
    # Crear secuencias
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    
    X_seq, y_seq = np.stack(X_seq), np.array(y_seq)
    
    # Dividir en train/val (80/20)
    val_size = max(1, int(0.2 * len(X_seq)))  # Asegurar al menos 1 muestra
    X_train, X_val = X_seq[:-val_size], X_seq[-val_size:]
    y_train, y_val = y_seq[:-val_size], y_seq[-val_size:]
    
    # Escalar
    scaler_X = StandardScaler()
    n_s, seq_len, n_feat = X_train.shape
    X_train_sc = scaler_X.fit_transform(X_train.reshape(-1, n_feat)).reshape(n_s, seq_len, n_feat)
    
    n_s_val, _, _ = X_val.shape
    X_val_sc = scaler_X.transform(X_val.reshape(-1, n_feat)).reshape(n_s_val, seq_len, n_feat)
    
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_sc = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    
    # Datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_sc, dtype=torch.float32),
        torch.tensor(y_train_sc, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_sc, dtype=torch.float32),
        torch.tensor(y_val_sc, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Modelo
    model = ImprovedLSTM(n_feat).to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Entrenamiento con early stopping
    best_val_loss = float('inf')
    patience, patience_counter = 20, 0
    train_losses, val_losses = [], []
    
    # INICIALIZAR best_model ANTES DEL BUCLE
    best_model = model.state_dict()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
        
        # Pérdidas promedio
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()  # actualizar el mejor modelo
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Cargar el mejor modelo encontrado durante el entrenamiento
    model.load_state_dict(best_model)
    
    return model, scaler_X, scaler_y, dfp, train_losses, val_losses, key_features

def forecast(model, scaler_X, scaler_y, dfp, lookback, h, key_features):
    # Usar solo características clave disponibles
    X = dfp[key_features].values
    window = X[-lookback:].copy()
    preds = []
    
    last_date = dfp['fecha'].iloc[-1]
    
    for i in range(1, h+1):
        next_date = last_date + pd.DateOffset(months=i)
        next_month = next_date.month
        next_year = next_date.year
        
        # Crear nueva fila de características
        new_row = []
        
        # Añadir características temporales
        if 'mes_sin' in key_features:
            new_row.append(np.sin(2 * np.pi * next_month / 12))
        if 'mes_cos' in key_features:
            new_row.append(np.cos(2 * np.pi * next_month / 12))
        if 'year_scaled' in key_features:
            min_year = dfp['year'].min()
            max_year = dfp['year'].max()
            year_range = max(1, max_year - min_year)  # Evitar división por cero
            new_row.append((next_year - min_year) / year_range)
        
        # Añadir características de tráfico con el último valor disponible
        if 'trafico_ligero_estimado' in key_features:
            traf_idx = key_features.index('trafico_ligero_estimado')
            new_row.append(window[-1, traf_idx])
        if 'trafico_pesado_estimado' in key_features:
            traf_idx = key_features.index('trafico_pesado_estimado')
            new_row.append(window[-1, traf_idx])
        
        # Si no hay características, usar valor cero
        if not new_row:
            new_row = [0] * len(key_features)
        
        window = np.vstack([window[1:], new_row])
        
        scaled_window = scaler_X.transform(window).reshape(1, lookback, -1)
        with torch.no_grad():
            p_sc = model(torch.tensor(scaled_window, dtype=torch.float32)).cpu().numpy()
        
        p = scaler_y.inverse_transform(p_sc)[0,0]
        preds.append(max(0, p))  # Asegurar valores no negativos
    
    return preds

# ==============================================
# 4. Interfaz de Usuario
# ==============================================
# Mapeo de provincias
code_to_name = {
    1: "Álava", 2: "Albacete", 3: "Alicante", 4: "Almería", 5: "Ávila",
    6: "Badajoz", 7: "Islas Baleares", 8: "Barcelona", 9: "Burgos",
    10: "Cáceres", 11: "Cádiz", 12: "Castellón", 13: "Ciudad Real",
    14: "Córdoba", 15: "A Coruña", 16: "Cuenca", 17: "Girona",
    18: "Granada", 19: "Guadalajara", 20: "Guipúzcoa", 21: "Huelva",
    22: "Huesca", 23: "Jaén", 24: "León", 25: "Lleida", 26: "La Rioja",
    27: "Lugo", 28: "Madrid", 29: "Málaga", 30: "Murcia", 31: "Navarra",
    32: "Ourense", 33: "Asturias", 34: "Palencia", 35: "Las Palmas",
    36: "Pontevedra", 37: "Salamanca", 38: "Santa Cruz de Tenerife",
    39: "Cantabria", 40: "Segovia", 41: "Sevilla", 42: "Soria",
    43: "Tarragona", 44: "Teruel", 45: "Toledo", 46: "Valencia",
    47: "Valladolid", 48: "Vizcaya", 49: "Zamora", 50: "Zaragoza",
    51: "Ceuta", 52: "Melilla"
}
name_to_code = {v.casefold(): k for k, v in code_to_name.items()}

def main():
    st.sidebar.header("Configuración del Modelo")
    
    # Selector de provincia
    provincia = st.sidebar.selectbox("Selecciona Provincia", sorted(code_to_name.values()))
    prov_code = name_to_code[provincia.lower()]
    
    # Parámetros del modelo
    lookback = st.sidebar.slider("Meses de histórico a considerar", 3, 9, 6)
    horizon = st.sidebar.slider("Meses a predecir", 1, 12, 6)
    
    # Mostrar datos preprocesados
    if st.sidebar.checkbox("Mostrar datos preprocesados"):
        st.subheader("Datos Preprocesados")
        st.dataframe(df_full.head(20))
    
    # Entrenar modelo y generar pronóstico
    if st.sidebar.button("Generar Pronóstico"):
        with st.spinner(f"Entrenando modelo para {provincia}..."):
            try:
                model, scaler_X, scaler_y, dfp, train_loss, val_loss, used_features = train_province(
                    df_full, prov_code, lookback, KEY_FEATURES)
            
                # Mostrar curva de aprendizaje
                st.subheader("Curva de Aprendizaje")
                fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
                ax_loss.plot(train_loss, label='Entrenamiento')
                ax_loss.plot(val_loss, label='Validación')
                ax_loss.set_title('Pérdida durante el Entrenamiento')
                ax_loss.set_ylabel('MSE')
                ax_loss.set_xlabel('Época')
                ax_loss.legend()
                ax_loss.grid(True)
                st.pyplot(fig_loss)
                
                # Generar pronóstico
                preds = forecast(model, scaler_X, scaler_y, dfp, lookback, horizon, used_features)
                
                # Visualización
                st.subheader(f"Pronóstico de Accidentes para {provincia}")
                
                # Preparar datos para el gráfico
                history = dfp[['fecha', 'NUM_ACCIDENTE_MES']].iloc[-24:]  # Últimos 2 años
                future_dates = [dfp['fecha'].iloc[-1] + pd.DateOffset(months=i) for i in range(1, horizon+1)]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Histórico
                ax.plot(history['fecha'], history['NUM_ACCIDENTE_MES'], 
                        'o-', color='blue', label='Histórico')
                
                # Pronóstico
                ax.plot(future_dates, preds, 's--', color='red', label='Pronóstico')
                
                # Conexión
                if not history.empty:
                    ax.plot([history['fecha'].iloc[-1], future_dates[0]], 
                            [history['NUM_ACCIDENTE_MES'].iloc[-1], preds[0]], 
                            ':', color='gray')
                
                ax.set_title(f"Pronóstico para los próximos {horizon} meses")
                ax.set_ylabel("Número de Accidentes")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Tabla de resultados
                st.subheader("Detalle del Pronóstico")
                forecast_data = {
                    'Fecha': future_dates,
                    'Accidentes Esperados': [int(round(p)) for p in preds]
                }
                forecast_df = pd.DataFrame(forecast_data)
                st.dataframe(forecast_df.style.format({'Fecha': lambda x: x.strftime('%Y-%m-%d')}))
                
            except ValueError as e:
                st.error(str(e))
    
    # Mostrar datos históricos
    if st.sidebar.checkbox("Mostrar datos históricos"):
        st.subheader(f"Datos Históricos para {provincia}")
        hist_data = df_full[df_full['COD_PROVINCIA'] == prov_code][['fecha', 'NUM_ACCIDENTE_MES']]
        st.line_chart(hist_data.set_index('fecha'))

if __name__ == '__main__':
    main()