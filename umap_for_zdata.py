import h5py
import pandas as pd
import umap
import plotly.express as px
from PIL import Image
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from PIL import Image
from io import BytesIO
import base64


def numpyTob64(array):
    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64

def getGravitySpyDatasetIndex(hoverData):
    label = hoverData['points'][0]['curveNumber']
    num = hoverData['points'][0]['pointNumber']
    return sum(each_label_data_num[:label]) + num

# datasetをpdDataFrameに格納
dataset_path = './data/trainingset.h5'
label_indies = pd.read_csv('./data/gravity_spy_labels.csv', index_col=0, header=None)
iic_pred_labels = pd.read_csv('./data/sc_pred_labels.csv', index_col=0, header=None)
z_autoencoder = pd.read_csv('./data/z-autoencoder-outputs.csv', index_col=0)


z_umap = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=None).fit_transform(z_autoencoder)
col_name = ['umap-component-1', 'umap-component-2', 'umap-component-3']
# 3d scatterの引数に対応したpdDataFrameに格納
df = pd.DataFrame(z_umap, index=z_autoencoder.index, columns=col_name)

# 各クラスのデータ数を取得, マウスホバー時に該当するデータを探査する際に使用
# [328, 232, 58, 1869, 66, 454, 279, 830, 573, 657, 453, 181, 88, 27, 453, 285, 459, 354, 116, 472, 44, 305]
each_label_data_num = z_autoencoder.index.value_counts(sort=False).values.tolist() # sortすると降順になってしまう


fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2],
                    color=df.index)
fig.update_traces(marker_size=3)
fig.update_layout(height=900)

app = dash.Dash(__name__)
app.layout = html.Div([
                       html.Div(id="output"),
                       dcc.Graph(id="image", figure=fig)
])

# マウスホバーしたデータを画像表示する関数をcallbackとして登録
@app.callback(Output('output', 'children'), [Input('image', 'hoverData')])
def displayImage(hoverData):
    if hoverData:
        idx = getGravitySpyDatasetIndex(hoverData)
        target_image = h5_handle[all_image_paths[idx]][0] # [0, 1, 2, 3] -> 0.5, 1.0, 2.0, 4.0 sec
        im_b64 = numpyTob64(target_image)
        value = 'data:image/png;base64,{}'.format(im_b64)
        return html.Img(src=value, height='200px')
    return None

# h5の全てのファイルパスを取得
all_image_paths = []
with h5py.File(dataset_path) as f:
    f.visit(lambda key : all_image_paths.append(key) if isinstance(f[key], h5py.Dataset) else None)
h5_handle = h5py.File(dataset_path, 'r')

app.run_server(debug=True) # debug:エラー時などweb上で確認ができる
