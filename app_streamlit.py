import streamlit as st
import h5py
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"
st.set_page_config(layout="wide")


def calc_fft(
    data: npt.ArrayLike, sampling_rate: float
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    fft = 2 * np.abs(np.fft.rfft(data)) / np.shape(data)
    f = np.linspace(0, sampling_rate / 2, np.shape(fft)[0])
    return fft, f


def idrofono_to_dBuPa(amplitudes: npt.ArrayLike) -> npt.ArrayLike:
    return 20 * np.log10(2 * amplitudes / np.sqrt(8)) + 217


FILE = "dati_elaborati.h5"
ampiezze: dict[str, dict[str, npt.ArrayLike]] = dict()

with h5py.File(FILE, "r") as f:
    top_groups = [k for k, v in f.items() if isinstance(v, h5py.Group)]
    for group in top_groups:
        ampiezze[group] = dict(
            nato=f[group]["amplitudes"]["nato"][:],
            coh=f[group]["amplitudes"]["coh"][:],
            frequenze=f[group]["frequencies_Hz"][:],
            label=f[group].attrs["label"],
        )

st.title("04/11/2025 - Analisi Idrofono NATO")
st.text("Sono state testate tre differenti teste sensore in fibra ottica:")
st.markdown(
    "- piastra sottile: un foglio di fibra di carbonio, plessibile, con 10 metri di fibra avvolta e incollata."
)
st.markdown(
    "- piastra spessa: foglio di fibra di carbonio, come sopra, con stessa lunghezza di fibra, ma con anche uno strato di rinforzo che ha reso la piastra rigida e coperto interamente la fibra."
)
st.markdown(
    "- rocchetto di fibra tight, buffer 900um, avvolta per un totale di 50 metri."
)
st.text(
    "Sono state tutte testate con interrogatore coerente, modificato per utilizzare laser Ioptis (quello del DAS) che ha una larghezza di righa di circa 1.5 kHz. Segnali acquisiti con scheda NI a 24 bit, 50 kS/s."
)
st.text(
    "La fibra sensore e gli specchi Faraday sono stati tenuti fuori vasca. La testa sensore immersa in vasca, vicino al generatore di segnale e all'idrofono di riferimento."
)
st.text(
    "Per ogni sensore sono stati generati toni a frequenze diverse, da 300 Hz a 5000 kHz, passo 100 Hz, con ampiezza costante. Per ogni frequenza sono stati generati vari burst di sinisoide finestrata."
)

st.subheader("Ampiezza vs Frequenza")
st.markdown(
    """Le ampiezze nei prossimi grafici sono state ottenute misurando l'ampiezza
    del picco alla frequenza in esame nella FFT del segnale.
    Nel caso dei dati dell'idrofono NATO è stata fatta la conversione a $\\frac{dB}{\mu Pa}$, tramite formula fornita:"""
)

st.latex(r"20*\log_{10}\frac{2*amplitude[V]}{\sqrt{8}} + 217")
st.markdown(
    """ 
    Invece per le misure del sistema in fibra otica non è stata fatta nessuna conversione e i valori sono ancora in radianti."""
)

for group in ampiezze.keys():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=ampiezze[group]["frequenze"],
            y=idrofono_to_dBuPa(ampiezze[group]["nato"]),
            name="Nato",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ampiezze[group]["frequenze"], y=ampiezze[group]["coh"], name="Cohaerentia"
        ),
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text="Ampiezza Cohaerentia (rad)",
        title_font_color="red",
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text="Ampiezza Nato (dB/1µPa)", title_font_color="blue", secondary_y=False
    )
    fig.update_xaxes(title_text="Frequenza (Hz)")
    fig.update_layout(
        title_text=f"<b>{ampiezze[group]['label']}</b> - Confronto ampiezze tono - Cohaerentia vs Nato"
    )
    st.plotly_chart(fig, theme=None)

st.markdown(
    """Come recap, in un singolo grafico le risposte delle tre differenti teste."""
)

fig_coh = go.Figure(
    layout=dict(
        title="<b>Confronto elementi Cohaerentia</b>",
        xaxis_title="Frequenza (Hz)",
        yaxis_title="Ampiezza (rad)",
    ),
    data=[
        go.Scatter(
            name=ampiezze[group]["label"],
            x=ampiezze[group]["frequenze"],
            y=ampiezze[group]["coh"],
        )
        for group in top_groups
    ],
)

st.plotly_chart(fig_coh, theme=None)

st.header("FFT")
st.text(
    "Selezionare la testa sensore di cui visualizzare le FFT per ogni frequenza. Le FFT sono normalizzate in base all'ampiezza di picco del tono generato in quel momento."
)


def plot_fft(head: str):
    fig_fft = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=("Nato", "Cohaerentia"),
    )
    with h5py.File(FILE, "r") as f:
        freqs = f[head]["frequencies_Hz"][:]
        raw_coh = f[head]["raw"]["coh"][:]
        raw_nato = f[head]["raw"]["nato"][:]
        color_scale = px.colors.sample_colorscale("thermal", len(freqs))

        for i, freq in enumerate(freqs):
            coh_avg_fft, coh_avg_f = calc_fft(raw_coh[i, :], 50_000)
            freq_index_coh = np.argmin(np.abs(coh_avg_f - float(freq)))
            max_near_freq_coh = np.max(
                coh_avg_fft[freq_index_coh - 10 : freq_index_coh + 11]
            )

            nato_fft, nato_f = calc_fft(raw_nato[i, :], 50_000)
            freq_index_nato = np.argmin(np.abs(nato_f - float(freq)))
            max_near_freq_nato = np.max(
                nato_fft[freq_index_nato - 10 : freq_index_nato + 11]
            )

            fig_fft.add_trace(
                go.Scatter(
                    x=nato_f,
                    y=nato_fft / max_near_freq_nato,
                    line_color=color_scale[i],
                    name=f"{freq} Hz",
                    legendgroup=f"{freq}",
                ),
                row=1,
                col=1,
            )
            fig_fft.add_trace(
                go.Scatter(
                    x=coh_avg_f,
                    y=coh_avg_fft / max_near_freq_coh,
                    line_color=color_scale[i],
                    legendgroup=f"{freq}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig_fft.update_xaxes(title_text="Frequency (Hz)")
        fig_fft.update_yaxes(title_text="Normalized amplitude", type="log")
        fig_fft.update_layout(
            title_text=f"<b>{ampiezze[head]['label']}</b> - FFT Normalizzate - Cohaerentia vs Nato",
            height=1000,
        )
        st.plotly_chart(fig_fft, theme=None)


st.selectbox(
    "Seleziona testa sensore:",
    options=list(ampiezze.keys()),
    key="fft_head",
)

plot_fft(st.session_state.fft_head)

st.header("Dati grezzi")

st.markdown(
    """Le FFT sopra sono state calcolate a partire dai dati grezzi finestrati e aggregati,
        ovvero mediando le generazioni alla stessa frequenza.
        Nel caso dei dati dell'idrofono NATO questo è stato fatto in automatico dalla loro strumentazione,
        visto che l'acquisizione era triggerata dal generatore di sinusoide.
        Nel caso dei dati del sensore in fibra ottica, la ricerca degli instanti temporali è stata fatta manualmente."""
)

st.selectbox(
    "Seleziona testa sensore:",
    options=list(ampiezze.keys()),
    key="raw_head",
)

freqs = ampiezze[st.session_state.raw_head]["frequenze"]
selected_freq = st.selectbox(
    "Seleziona frequenza da visualizzare:",
    options=freqs,
    key="raw_freq",
)


def plot_raw(head: str, frequency: str):
    with h5py.File(FILE, "r") as f:
        index = np.where(f[head]["frequencies_Hz"][:] == float(frequency))[0][0]
        nato_raw = f[head]["raw"]["nato"][index, :]
        coh_raw = f[head]["raw"]["coh"][index, :]
        time = np.arange(nato_raw.shape[0]) / 50_000

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=time - 0.00028,
                y=nato_raw,
                name="Nato",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=coh_raw,
                name="Cohaerentia",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Ampiezza Cohaerentia (rad)",
            title_font_color="red",
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Ampiezza Nato (V)", title_font_color="blue", secondary_y=False
        )
        fig.update_xaxes(title_text="Tempo (s)")
        fig.update_layout(
            title_text=f"<b>{ampiezze[head]['label']}</b> - Frequenza {frequency} Hz"
        )
        st.plotly_chart(fig, theme=None)


plot_raw(st.session_state.raw_head, st.session_state.raw_freq)
