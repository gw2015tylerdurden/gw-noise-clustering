import base64
from io import BytesIO
import numpy as np
import dash
import h5py
import pandas as pd
import plotly.express as px
import umap
from dash import dcc, html
from dash.dependencies import Input, Output
from PIL import Image
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist

def numpyTob64FmtPng(array):
    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64

random_state = 123
nclass = 23
umap_min_dist = 0.4
umap_neighbor = 15
is_selftuning = True

# random_state = 123
# nclass = 17
# umap_min_dist = 0.2
# umap_neighbor = 15
# is_selftuning = False  # median heuristics

# datasetをpdDataFrameに格納
dataset_path = './data/trainingset.h5'
label_indies = pd.read_csv('./data/gravity_spy_labels.csv', index_col=0, header=None)
iic_pred_labels = pd.read_csv('./data/sc_pred_labels.csv', index_col=0, header=None)
z_autoencoder = pd.read_csv('./data/z-autoencoder-outputs.csv', index_col=0)

z_umap = umap.UMAP(n_components=3, n_neighbors=umap_neighbor, min_dist=umap_min_dist, random_state=random_state).fit_transform(z_autoencoder)

col_name = ['umap-component-1', 'umap-component-2', 'umap-component-3']
# 3d scatterの引数に対応したpdDataFrameに格納
df = pd.DataFrame(z_umap, index=z_autoencoder.index, columns=col_name)


if is_selftuning:
    from src.utils import calc_selfturining_affinity
    X, _ = calc_selfturining_affinity(z_umap, 15)
    sc = SpectralClustering(n_clusters=nclass, random_state=random_state, assign_labels="kmeans", affinity='precomputed').fit(X)
    filename = f"sc_labels_random_state{random_state}_nclass{nclass}_umap_neighbor{umap_neighbor}_umap_min_dist{umap_min_dist}_selfturning.csv"
else:
    gamma = 2.0 / np.median(pdist(z_umap)) ** 2
    sc = SpectralClustering(n_clusters=nclass, random_state=random_state, assign_labels="kmeans", gamma=gamma).fit(z_umap)
    filename = f"sc_labels_random_state{random_state}_nclass{nclass}_umap_neighbor{umap_neighbor}_umap_min_dist{umap_min_dist}_gamma{gamma}.csv"

# 各クラスのデータ数を取得, マウスホバー時に該当するデータを探査する際に使用
sc_labels = pd.DataFrame(sc.labels_, columns=['sc_labels'])
sc_labels.to_csv(filename)

df['customdata'] = list(range(len(df.index)))

# default表示
fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2], color=df.index, hover_data=['customdata'])

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id="image", figure=fig),
        html.Div([
            dcc.Checklist(
                id='color-toggle',
                options=[{'label': 'Spectral Clustering Labels', 'value': 'index'}],
                value=[],
                inputStyle={"margin-right": "5px", "margin-left": "5px"}
            )
        ], style={'display': 'inline-block'})
    ]),
    html.Div(id="output"),
])

# 現在のカメラの状態を格納するグローバル変数
current_camera = None

# カメラの状態を更新する関数
def update_camera(camera):
    global current_camera
    current_camera = camera

# Define the new callback triggered by the button
@app.callback(
    Output('image', 'figure'),  # Output is the figure of the dcc.Graph component
    Input('color-toggle', 'value'),  # Input is the number of times the button is clicked
    Input('image', 'relayoutData')  # Input is the relayoutData of the dcc.Graph component
)
def switch_color(value, relayout_data):
    global current_camera

    # カメラの状態が更新された場合
    if relayout_data and 'scene.camera' in relayout_data:
        update_camera(relayout_data['scene.camera'])

    if 'index' in value:
        # If the toggle button is checked, switch the color parameter to sc.labels_
        #new_fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2], color=sc_labels['sc_labels'], hover_name=z_autoencoder.index)
        new_fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2], color=sc_labels['sc_labels'].astype(str), hover_name=z_autoencoder.index, color_discrete_sequence=px.colors.sequential.Viridis, hover_data=['customdata'])
    else:
        # If the toggle button is unchecked, switch the color parameter back to df.index
        new_fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2], color=df.index, hover_name=z_autoencoder.index, hover_data=['customdata'])


    new_fig.update_traces(marker_size=1, mode='markers', marker=dict(showscale=False))
    # fix Graph plot
    new_fig.update_layout(legend=dict(itemsizing='constant'))
    new_fig.update_layout(
        height=900,
        font=dict(size=18),
        legend=dict(font=dict(size=20))
    )

    # 現在のカメラの状態を新しいフィギュアに適用
    if current_camera:
       new_fig.update_layout(scene_camera=current_camera)

    return new_fig


# マウスホバーしたデータを画像表示する関数をcallbackとして登録
@app.callback(Output('output', 'children'), [Input('image', 'hoverData')])
def displayImage(hoverData):
    if hoverData:
        idx = hoverData['points'][0]['customdata'][0]
        images = [h5_handle[all_image_paths[idx]][i] for i in range(4)] # [0, 1, 2, 3] -> 0.5, 1.0, 2.0, 4.0 sec
        spacer = np.zeros((images[0].shape[0], 10)).astype(np.uint8)  # 10ピクセル分の隙間を作成する, uint8はpng変換で必要
        target_image = np.concatenate([images[0], spacer, images[1], spacer, images[2], spacer, images[3]], axis=1)
        im_b64 = numpyTob64FmtPng(target_image)
        value = 'data:image/png;base64,{}'.format(im_b64)
        return html.Img(src=value, height='200px')
    return None

# h5の全てのファイルパスを取得
all_image_paths = []
with h5py.File(dataset_path) as f:
    f.visit(lambda key : all_image_paths.append(key) if isinstance(f[key], h5py.Dataset) else None)
h5_handle = h5py.File(dataset_path, 'r')

app.run_server(debug=True) # debug:エラー時などweb上で確認ができる
