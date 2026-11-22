use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct UnifiedModelConfig {
    pub vocab_size: usize,
    pub blank_id: usize,
    pub decoder_lstm_dim: usize,
    pub decoder_lstm_layers: usize,
    pub subsampling_factor: usize,
}

impl Default for UnifiedModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1025,
            blank_id: 1024,
            decoder_lstm_dim: 640,
            decoder_lstm_layers: 2,
            subsampling_factor: 8,
        }
    }
}

pub struct ParakeetUnifiedModel {
    encoder: Session,
    decoder_joint: Session,
    pub config: UnifiedModelConfig,
}

impl ParakeetUnifiedModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
        config: UnifiedModelConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_joint_path = Self::find_decoder_joint(model_dir)?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_joint = builder.commit_from_file(&decoder_joint_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
            config,
        })
    }

    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        let candidates = ["encoder.onnx", "encoder.int8.onnx", "encoder-model.onnx"];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(Error::Config(format!(
            "No unified encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder_joint(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "decoder_joint.onnx",
            "decoder_joint.int8.onnx",
            "decoder_joint-model.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(Error::Config(format!(
            "No unified decoder_joint model found in {}",
            dir.display()
        )))
    }

    pub fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        let input = features
            .t()
            .to_shape((1, feature_size, time_steps))
            .map_err(|e| Error::Model(format!("Failed to build encoder input: {e}")))?
            .to_owned();

        let input_length = Array1::from_vec(vec![time_steps as i64]);

        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => ort::value::Value::from_array(input)?,
            "length" => ort::value::Value::from_array(input_length)?
        ))?;

        let (shape, data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {e}")))?;

        let (_, lens_data) = outputs["encoded_lengths"]
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder lengths: {e}")))?;

        let dims = shape.as_ref();
        if dims.len() != 3 {
            return Err(Error::Model(format!(
                "Expected 3D encoder output, got shape: {dims:?}"
            )));
        }

        let encoder_out = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create encoder array: {e}")))?;

        Ok((encoder_out, lens_data[0]))
    }

    /// One decoder_joint run over encoder `frames` `(1, hidden, T)` for each
    /// target sequence in `targets`, all starting from the same predictor
    /// state. Each sequence starts with the token the predictor consumes
    /// first (the previous token, or blank at the start). Returns the joint
    /// logits, laid out `[batch][frame][target position][vocab]`, and their
    /// shape.
    pub fn run_joint_batch(
        &mut self,
        frames: &Array3<f32>,
        targets: &[Vec<i32>],
        state_1: &Array3<f32>,
        state_2: &Array3<f32>,
    ) -> Result<(Vec<f32>, [usize; 4])> {
        let batch = targets.len();
        let (hidden, time) = (frames.shape()[1], frames.shape()[2]);
        let max_len = targets.iter().map(Vec::len).max().unwrap_or(0);
        if batch == 0 || max_len == 0 || time == 0 {
            return Err(Error::Model("run_joint_batch: empty input".into()));
        }
        let encoder = frames
            .broadcast((batch, hidden, time))
            .ok_or_else(|| Error::Model("run_joint_batch: bad frame shape".into()))?
            .to_owned();
        let pad = self.config.blank_id as i32;
        let padded = Array2::from_shape_fn((batch, max_len), |(b, u)| {
            targets[b].get(u).copied().unwrap_or(pad)
        });
        let lengths = Array1::from_iter(targets.iter().map(|t| t.len() as i32));
        let repeat = |state: &Array3<f32>| -> Result<Array3<f32>> {
            let (layers, dim) = (state.shape()[0], state.shape()[2]);
            Ok(state
                .broadcast((layers, batch, dim))
                .ok_or_else(|| Error::Model("run_joint_batch: bad state shape".into()))?
                .to_owned())
        };

        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => ort::value::Value::from_array(encoder)?,
            "targets" => ort::value::Value::from_array(padded)?,
            "target_length" => ort::value::Value::from_array(lengths)?,
            "input_states_1" => ort::value::Value::from_array(repeat(state_1)?)?,
            "input_states_2" => ort::value::Value::from_array(repeat(state_2)?)?
        ])?;
        let (shape, data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;
        let dims = shape.as_ref();
        if dims.len() != 4 {
            return Err(Error::Model(format!("Expected 4D joint output, got {dims:?}")));
        }
        let dims = [dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize];
        Ok((data.to_vec(), dims))
    }

    pub fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>,
        target_token: i32,
        state_1: &Array3<f32>,
        state_2: &Array3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        let targets = Array2::from_elem((1, 1), target_token);
        let target_length = Array1::from_elem(1, 1i32);

        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => ort::value::Value::from_array(encoder_frame.clone())?,
            "targets" => ort::value::Value::from_array(targets)?,
            "target_length" => ort::value::Value::from_array(target_length)?,
            "input_states_1" => ort::value::Value::from_array(state_1.clone())?,
            "input_states_2" => ort::value::Value::from_array(state_2.clone())?
        ])?;

        let (_, logits_data) = outputs["outputs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;

        let logits = Array1::from_vec(logits_data.to_vec());

        let (h_shape, h_data) = outputs["output_states_1"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state_1: {e}")))?;
        let (c_shape, c_data) = outputs["output_states_2"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract state_2: {e}")))?;

        let new_state_1 = Array3::from_shape_vec(
            (
                h_shape[0] as usize,
                h_shape[1] as usize,
                h_shape[2] as usize,
            ),
            h_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape state_1: {e}")))?;

        let new_state_2 = Array3::from_shape_vec(
            (
                c_shape[0] as usize,
                c_shape[1] as usize,
                c_shape[2] as usize,
            ),
            c_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape state_2: {e}")))?;

        Ok((logits, new_state_1, new_state_2))
    }
}
