use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateJob {
    pub source_lang: String,
    pub target_lang: String,
    pub context: Option<String>,
    pub input: TranslateInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranslateInput {
    Text(String),
    /// 파일 경로 (text or .json). 출력 경로 지정 없으면 <in>.<tgt>.* 로 저장.
    File { path: String, out: Option<String> },
    /// 리스트 (쉼표 분리)
    List(Vec<String>),
}

impl TranslateJob {
    pub fn display_label(&self) -> String {
        match &self.input {
            TranslateInput::Text(t) => {
                let t = t.replace('\n', " ");
                if t.chars().count() > 40 {
                    format!("\"{}…\"", t.chars().take(40).collect::<String>())
                } else {
                    format!("\"{t}\"")
                }
            }
            TranslateInput::File { path, .. } => format!("file:{path}"),
            TranslateInput::List(items) => format!("list({})", items.len()),
        }
    }
}
