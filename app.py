from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from bsb_bvh_process import bvh_to_dataframe, angle_between_vectors_from_three_points
from basketball_analysis import process_basketball_data, find_wrist_head_threshold_indices, find_all_wrist_head_threshold_intervals,  find_max_angle_velocity_indices
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting BVH 3D application...")

# functions


def plot_3d_points(df, point_cols, index):
    """
    Visualize a list of 3D points from df at a given frame index using Plotly.
    Parameters:
        df: pandas DataFrame with columns for coordinates
        point_cols: list of [x_col, y_col, z_col] for each point (e.g. [['Hips_X','Hips_Y','Hips_Z'], ...])
        index: frame index (int) to visualize
    """
    xs, ys, zs = [], [], []
    labels = []
    for cols in point_cols:
        x, y, z = df.loc[index, cols]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        labels.append(cols[0].rsplit('_', 1)[0])  # Use joint name as label

    # Calculate the full range for all axes across all frames and selected points
    x_all, y_all, z_all = [], [], []
    for cols in point_cols:
        x_all.extend(df[cols[0]].values)
        y_all.extend(df[cols[1]].values)
        z_all.extend(df[cols[2]].values)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers+text',
            marker=dict(size=6, color='blue'),
            text=labels,
            textposition='top center',
            line=dict(color='gray', width=2)
        )
    ])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
#            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[0, 2.5]),
#            zaxis=dict(range=[z_min, z_max]),
            aspectmode='data',  # keeps proportions correct
            camera=dict(
                eye=dict(x=0, y=0, z=2.0),
                up=dict(x=0, y=1, z=0)
            )
        ),
        title=f'3D Skeleton Points at frame: #{index}',
        width=400,
        height=800,
        autosize=False,
        margin=dict(l=0, r=0, b=0, t=20)
    )
    print(f"Generating 3D plot for frame {index}")
    return fig

def plot_all_data(df):
    fig4 = make_subplots(rows=2, cols=1)

    fig4.add_trace(go.Scatter(x=df.index, y=df['Head_Y'], mode='lines', name='Head '), row=1, col=1)

    fig4.add_trace(go.Scatter(x=df.index, y=df['LeftHand_Y'], mode='lines', name='L hand vertical coordinate'), row=1, col=1)

    fig4.add_trace(go.Scatter(x=df.index, y=df['RightHand_Y'], mode='lines', name='R hand vertical coordinate'), row=1, col=1)

    intervals = find_all_wrist_head_threshold_intervals(df, ['LeftHand_Y','RightHand_Y','Head_Y'])

    for i in intervals:

        START_idx, STOP_idx = i

        print(START_idx, STOP_idx)

        df_ = df.loc[START_idx:STOP_idx]
        
        R_V_max_idx, L_V_max_idx = find_max_angle_velocity_indices(df_, ['R_elbow_V', 'L_elbow_V'])
        
        cols = ['LeftHand_Y','RightHand_Y','Head_Y']
        
        # get min/max for each column
        stats = df_[cols].agg(['min', 'max'])
        
        overall_min = df_[cols].values.min()
        overall_max = df_[cols].values.max()
        
        fig4.add_shape(
                type="line",
                x0=START_idx,
                y0=overall_min,
                x1=START_idx,
                y1=overall_max,
                xref='x1',  # Reference to the x-axis of the third subplot
                yref='y1',  # Reference to the y-axis of the third subplot
                line=dict(
                    color="Black",
                    width=1,
                    dash="dash",
                ),
            )
        
        fig4.add_shape(
                type="line",
                x0=STOP_idx,
                y0=overall_min,
                x1=STOP_idx,
                y1=overall_max,
                xref='x1',  # Reference to the x-axis of the third subplot
                yref='y1',  # Reference to the y-axis of the third subplot
                line=dict(
                    color="Black",
                    width=1,
                    dash="dash",
                ),
            )

        cols = ['R_elbow_V', 'L_elbow_V']
        
        # get min/max for each column
        stats = df_[cols].agg(['min', 'max'])
        
        overall_min = df_[cols].values.min()
        overall_max = df_[cols].values.max()
        
        fig4.add_shape(
                type="line",
                x0=START_idx,
                y0=overall_min,
                x1=START_idx,
                y1=overall_max,
                xref='x2',  
                yref='y2',  
                line=dict(
                    color="Black",
                    width=1,
                    dash="dash",
                ),
            )
        
        fig4.add_shape(
                type="line",
                x0=STOP_idx,
                y0=overall_min,
                x1=STOP_idx,
                y1=overall_max,
                xref='x2', 
                yref='y2',  
                line=dict(
                    color="Black",
                    width=1,
                    dash="dash",
                ),
            )

        if df['R_elbow_V'].iloc[R_V_max_idx] >= df['L_elbow_V'].iloc[L_V_max_idx]:
            y_val = df['R_elbow_V'].iloc[R_V_max_idx]
            x_val = R_V_max_idx
        else:
            y_val = df['L_elbow_V'].iloc[L_V_max_idx]
            x_val = L_V_max_idx
        # Add a red marker on subplot 2 (row=2)
        fig4.add_trace(
            go.Scatter(
                x=[x_val],
                y=[y_val],
                mode='markers',
                marker=dict(color='red', size=12),
                name='Ball take-off time'
            ),
            row=2, col=1
        )


    fig4.add_trace(go.Scatter(x=df.index, y=df['R_elbow_V'], mode='lines', name='R elbow angle velocity'), row=2, col=1)

    fig4.add_trace(go.Scatter(x=df.index, y=df['L_elbow_V'], mode='lines', name='L elbow angle velocity'), row=2, col=1)

    fig4.update_layout(showlegend=True)

    fig4.update_layout(
        autosize=False,
        width=1000,
        height=800,
    )

    #fig4.show()
    return fig4

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'bvh'}

logger.info("Creating Flask application...")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Basic configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
logger.info("Flask app configured successfully")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint for deployment verification"""
    return jsonify({"status": "healthy", "message": "BVH 3D app is running"}), 200

@app.route('/')
def home():
    # Render the home page without any initial plot
    return render_template('home.html', plot_html='')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("File upload request received")
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved: {filepath}")
            
            df = bvh_to_dataframe(filepath)
            app.config['last_df'] = df  # store DataFrame for later frame plotting

            df['R_elbow'] = angle_between_vectors_from_three_points(
            df,
            ['RightHand_X', 'RightHand_Y', 'RightHand_Z'],
            ['RightForeArm_X', 'RightForeArm_Y', 'RightForeArm_Z'],
            ['RightArm_X', 'RightArm_Y', 'RightArm_Z'],
            ['RightForeArm_X', 'RightForeArm_Y', 'RightForeArm_Z']
            )

            df['L_elbow'] = angle_between_vectors_from_three_points(
                df,
                ['LeftHand_X', 'LeftHand_Y', 'LeftHand_Z'],
                ['LeftForeArm_X', 'LeftForeArm_Y', 'LeftForeArm_Z'],
                ['LeftArm_X', 'LeftArm_Y', 'LeftArm_Z'],
                ['LeftForeArm_X', 'LeftForeArm_Y', 'LeftForeArm_Z']
            )

            df['R_elbow_V'] = df['R_elbow'].diff()

            df['L_elbow_V'] = df['L_elbow'].diff()
            # Fill NaN values in velocity columns
            df['R_elbow_V'] = df['R_elbow_V'].bfill()
            df['L_elbow_V'] = df['L_elbow_V'].bfill()

            # Generate the full analysis plot and return JSON for client-side Plotly
            fig = plot_all_data(df)
            # Convert figure to Plotly JSON and ensure no numpy arrays remain
            fig_json = fig.to_plotly_json()
            # Convert numpy arrays in data traces to lists
            for trace in fig_json.get('data', []):
                for coord in ('x', 'y', 'z', 'values'):  # include any array fields
                    if coord in trace and hasattr(trace[coord], 'tolist'):
                        trace[coord] = trace[coord].tolist()
            n_frames = len(df)
            # Prepare initial 3D frame plot (frame 0)
            point_cols = [
                ['RightHand_X', 'RightHand_Y', 'RightHand_Z'],
                ['RightForeArm_X', 'RightForeArm_Y', 'RightForeArm_Z'],
                ['RightArm_X', 'RightArm_Y', 'RightArm_Z'],
                ['Head_X', 'Head_Y', 'Head_Z'],
                ['LeftHand_X', 'LeftHand_Y', 'LeftHand_Z'],
                ['LeftForeArm_X', 'LeftForeArm_Y', 'LeftForeArm_Z'],
                ['LeftArm_X', 'LeftArm_Y', 'LeftArm_Z'],
                ['LeftUpLeg_X', 'LeftUpLeg_Y', 'LeftUpLeg_Z'],
                ['LeftLeg_X', 'LeftLeg_Y', 'LeftLeg_Z'],
                ['RightUpLeg_X', 'RightUpLeg_Y', 'RightUpLeg_Z'],
                ['RightLeg_X', 'RightLeg_Y', 'RightLeg_Z'],
                ['Hips_X', 'Hips_Y', 'Hips_Z'],
                ['RightFoot_X', 'RightFoot_Y', 'RightFoot_Z'],
                ['LeftFoot_X', 'LeftFoot_Y', 'LeftFoot_Z']
            ]
            initial_fig3 = plot_3d_points(df, point_cols, index=0)
            fig3_json = initial_fig3.to_plotly_json()
            # serialize numpy arrays in 3D plot
            for trace in fig3_json.get('data', []):
                for coord in ('x', 'y', 'z'):
                    if coord in trace and hasattr(trace[coord], 'tolist'):
                        trace[coord] = trace[coord].tolist()
            return jsonify({
                'success': True,
                'figure': fig_json,
                'n_frames': n_frames,
                'frame0': fig3_json
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/plot_data', methods=['POST'])
def plot_data():
    req = request.get_json()
    # Instead of using the slider, just plot all data using plot_all_data
    # You need to reconstruct the DataFrame from the JSON data
    data = req['data']
    # Reconstruct DataFrame from the joint data
    index = data['index']
    time = data['time']
    df = pd.DataFrame(index=index)
    df['Time'] = time

    for joint in data['joints']:
        df[joint] = data['joints'][joint]
    # If the user wants to see the full analysis plot, use plot_all_data
    fig = plot_all_data(df)
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id='plotly-all-data')
    return jsonify({'plot_html': plot_html})

# New endpoint for per-frame 3D point plot
@app.route('/frame_plot', methods=['POST'])
def frame_plot():
    req = request.get_json()
    try:
        idx = int(req.get('index'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid frame index'}), 400
    df = app.config.get('last_df')
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    # specify the joints to plot
    point_cols = [
        ['RightHand_X', 'RightHand_Y', 'RightHand_Z'],
        ['RightForeArm_X', 'RightForeArm_Y', 'RightForeArm_Z'],
        ['RightArm_X', 'RightArm_Y', 'RightArm_Z'],
        ['Head_X', 'Head_Y', 'Head_Z'],
        ['LeftHand_X', 'LeftHand_Y', 'LeftHand_Z'],
        ['LeftForeArm_X', 'LeftForeArm_Y', 'LeftForeArm_Z'],
        ['LeftArm_X', 'LeftArm_Y', 'LeftArm_Z'],
        ['LeftUpLeg_X', 'LeftUpLeg_Y', 'LeftUpLeg_Z'],
        ['LeftLeg_X', 'LeftLeg_Y', 'LeftLeg_Z'],
        ['RightUpLeg_X', 'RightUpLeg_Y', 'RightUpLeg_Z'],
        ['RightLeg_X', 'RightLeg_Y', 'RightLeg_Z'],
        ['Hips_X', 'Hips_Y', 'Hips_Z'],
        ['RightFoot_X', 'RightFoot_Y', 'RightFoot_Z'],
        ['LeftFoot_X', 'LeftFoot_Y', 'LeftFoot_Z'],
        ['RightToeBase_X', 'RightToeBase_Y', 'RightToeBase_Z'],
        ['LeftToeBase_X', 'LeftToeBase_Y', 'LeftToeBase_Z']
    ]
    fig = plot_3d_points(df, point_cols, index=idx)
    fig_json = fig.to_plotly_json()
    # convert numpy arrays in traces to lists
    for trace in fig_json.get('data', []):
        for coord in ('x', 'y', 'z'):
            if coord in trace and hasattr(trace[coord], 'tolist'):
                trace[coord] = trace[coord].tolist()
    return jsonify({'figure': fig_json})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on host=0.0.0.0, port={port}")
    app.run(host='0.0.0.0', port=port, debug=False)
