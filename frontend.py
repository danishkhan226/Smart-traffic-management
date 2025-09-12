from dash import Dash, dcc, html, Input, Output
import requests
import base64

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Smart Traffic Management Dashboard"),
    dcc.Upload(
        id='upload-video',
        children=html.Div(['Drag and Drop or ', html.A('Select Video File')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
        multiple=False
    ),
    html.Div(id='output-analysis')
])

@app.callback(
    Output('output-analysis', 'children'),
    Input('upload-video', 'contents')
)
def analyze_video(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        response = requests.post("http://localhost:8000/process_video/", files={'file': decoded})
        if response.status_code == 200:
            data = response.json()
            if 'error' in data:
                return html.Div(f"Error: {data['error']}")
            elif 'average_vehicles' in data and 'congestion_level' in data:
                return html.Div([
                    html.H3(f"Average Vehicles Detected: {data['average_vehicles']}"),
                    html.H3(f"Congestion Level: {data['congestion_level']}")
                ])
            else:
                return html.Div("Unexpected response format from backend")
        else:
            return html.Div(f"Error processing video: {response.status_code} - {response.text}")
    return html.Div("No video uploaded")

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)