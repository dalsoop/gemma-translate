use crate::app::{App, NewJobField, NewJobType};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame,
};

pub fn draw(f: &mut Frame, full: Rect, app: &App) {
    let w = full.width.min(80);
    let h = 18u16;
    let x = full.x + (full.width.saturating_sub(w)) / 2;
    let y = full.y + full.height.saturating_sub(h) / 2;
    let area = Rect { x, y, width: w, height: h };

    f.render_widget(Clear, area);
    let block = Block::default()
        .title(" 새 작업 추가 ")
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let form = &app.new_job;

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // type
            Constraint::Length(2), // src
            Constraint::Length(2), // tgt
            Constraint::Length(2), // main
            Constraint::Length(2), // extra
            Constraint::Min(1),    // help
        ])
        .split(inner);

    let typ_text = match form.job_type {
        NewJobType::Translate => "Translate  (< / > 로 변경)",
        NewJobType::Sentry => "Sentry i18n  (< / > 로 변경)",
    };
    static_field(f, rows[0], "종류  ", typ_text, form.focus == NewJobField::Type);

    input_field(f, rows[1], "src   ", form.src_lang.value(), form.focus == NewJobField::Src);
    input_field(f, rows[2], "tgt   ", form.tgt_lang.value(), form.focus == NewJobField::Tgt);

    match form.job_type {
        NewJobType::Translate => {
            input_field(f, rows[3], "text  ", form.text.value(), form.focus == NewJobField::Main);
            input_field(f, rows[4], "ctx   ", form.context.value(), form.focus == NewJobField::Extra);
        }
        NewJobType::Sentry => {
            static_field(f, rows[3], "step  ", form.sentry_step.as_str(), form.focus == NewJobField::Main);
            static_field(f, rows[4], "opts  ",
                if form.cache_bust { "--bust (b 로 토글)" } else { "(b 로 --bust 토글)" },
                form.focus == NewJobField::Extra);
        }
    }

    let hint = Line::from(vec![Span::styled(
        "[Tab/Shift-Tab] 이동   [1-6] step   [b] bust   [Enter] 추가   [Esc] 취소",
        Style::default().fg(Color::DarkGray),
    )]);
    f.render_widget(Paragraph::new(hint).alignment(Alignment::Center), rows[5]);
}

fn static_field(f: &mut Frame, area: Rect, label: &str, value: &str, focused: bool) {
    let style = if focused {
        Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };
    let line = Line::from(vec![
        Span::styled(format!(" {label} "), Style::default().fg(Color::Cyan)),
        Span::styled(format!(" {value} "), style),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn input_field(f: &mut Frame, area: Rect, label: &str, value: &str, focused: bool) {
    let cursor = if focused { "▏" } else { "" };
    static_field(f, area, label, &format!("{value}{cursor}"), focused);
}
