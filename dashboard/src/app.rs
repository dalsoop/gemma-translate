use crate::backend::gpu::GpuSnapshot;
use crate::config::Config;
use crate::jobs::{sentry::{SentryJob, SentryStep}, translate::{TranslateInput, TranslateJob}, Job, JobKind};
use std::collections::VecDeque;
use std::sync::Arc;
use tui_input::Input;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode { Normal, NewJob, Help }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus { Gpu, Jobs, History, Log }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewJobType { Translate, Sentry }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewJobField { Type, Src, Tgt, Main, Extra }

pub struct NewJobForm {
    pub focus: NewJobField,
    pub job_type: NewJobType,
    pub src_lang: Input,
    pub tgt_lang: Input,
    pub text: Input,
    pub context: Input,
    pub sentry_step: SentryStep,
    pub cache_bust: bool,
}

impl NewJobForm {
    pub fn new(cfg: &Config) -> Self {
        Self {
            focus: NewJobField::Type,
            job_type: NewJobType::Translate,
            src_lang: Input::from(cfg.defaults.source_lang.clone()),
            tgt_lang: Input::from(cfg.defaults.target_lang.clone()),
            text: Input::default(),
            context: Input::from(cfg.defaults.context.clone()),
            sentry_step: SentryStep::Sync,
            cache_bust: true,
        }
    }

    pub fn next_field(&mut self) {
        self.focus = match self.focus {
            NewJobField::Type => NewJobField::Src,
            NewJobField::Src => NewJobField::Tgt,
            NewJobField::Tgt => NewJobField::Main,
            NewJobField::Main => NewJobField::Extra,
            NewJobField::Extra => NewJobField::Type,
        };
    }

    pub fn prev_field(&mut self) {
        self.focus = match self.focus {
            NewJobField::Type => NewJobField::Extra,
            NewJobField::Src => NewJobField::Type,
            NewJobField::Tgt => NewJobField::Src,
            NewJobField::Main => NewJobField::Tgt,
            NewJobField::Extra => NewJobField::Main,
        };
    }

    /// 현재 편집 가능한 필드의 tui-input 참조 (편집 대상이 아닌 필드는 None)
    pub fn editable_input(&mut self) -> Option<&mut Input> {
        match (self.focus, self.job_type) {
            (NewJobField::Src, _) => Some(&mut self.src_lang),
            (NewJobField::Tgt, _) => Some(&mut self.tgt_lang),
            (NewJobField::Main, NewJobType::Translate) => Some(&mut self.text),
            (NewJobField::Extra, NewJobType::Translate) => Some(&mut self.context),
            _ => None,
        }
    }

    pub fn to_job(&self) -> Option<Job> {
        match self.job_type {
            NewJobType::Translate => {
                let text = self.text.value().to_string();
                if text.trim().is_empty() { return None; }
                let input = if std::path::Path::new(&text).exists() {
                    TranslateInput::File { path: text.clone(), out: None }
                } else {
                    TranslateInput::Text(text)
                };
                let ctx = self.context.value().trim().to_string();
                Some(Job::new(JobKind::Translate(TranslateJob {
                    source_lang: self.src_lang.value().to_string(),
                    target_lang: self.tgt_lang.value().to_string(),
                    context: if ctx.is_empty() { None } else { Some(ctx) },
                    input,
                })))
            }
            NewJobType::Sentry => Some(Job::new(JobKind::SentryI18n(SentryJob {
                step: self.sentry_step,
                cache_bust: self.cache_bust,
                workers: 32,
                sources: None,
                limit: None,
            }))),
        }
    }
}

pub struct App {
    pub cfg: Arc<Config>,
    pub mode: Mode,
    pub focus: Focus,
    pub new_job: NewJobForm,
    pub gpu: GpuSnapshot,
    pub active: Vec<Job>,
    pub queue: VecDeque<Job>,
    pub history: Vec<Job>,
    pub log: VecDeque<String>,
    pub selected_active: usize,
    pub should_quit: bool,
    pub active_connector: String,
    pub available_connectors: Vec<String>,
}

impl App {
    pub fn new(cfg: Arc<Config>) -> Self {
        let new_job = NewJobForm::new(&cfg);
        Self {
            cfg,
            mode: Mode::Normal,
            focus: Focus::Jobs,
            new_job,
            gpu: GpuSnapshot::default(),
            active: Vec::new(),
            queue: VecDeque::new(),
            history: Vec::new(),
            log: VecDeque::new(),
            selected_active: 0,
            should_quit: false,
            active_connector: "gemma".into(),
            available_connectors: vec!["gemma".into()],
        }
    }

    pub fn push_log(&mut self, line: String) {
        self.log.push_back(line);
        if self.log.len() > 500 { self.log.pop_front(); }
    }
}
