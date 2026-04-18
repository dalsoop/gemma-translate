use crate::app::App;
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame,
};

pub fn draw(f: &mut Frame, full: Rect, _app: &App) {
    let w = full.width.min(60);
    let h = 17u16;
    let x = full.x + (full.width.saturating_sub(w)) / 2;
    let y = full.y + full.height.saturating_sub(h) / 2;
    let area = Rect { x, y, width: w, height: h };

    f.render_widget(Clear, area);
    let block = Block::default()
        .title(" 도움말 ")
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(Color::Cyan));
    f.render_widget(block.clone(), area);

    let rows = vec![
        row("n", "새 작업 추가"),
        row("?", "이 도움말 열기"),
        row("Tab / Shift-Tab", "포커스 / 필드 이동"),
        row("↑ ↓", "실행중 목록 선택"),
        row("x", "선택된 실행중 작업 취소"),
        row("q / Ctrl-C", "종료"),
        Line::from(""),
        Line::styled(" ─ 새 작업 모달 ─ ", Style::default().fg(Color::DarkGray)),
        row("< / >", "Translate ↔ Sentry 전환"),
        row("1 ~ 6", "Sentry step 선택"),
        row("b", "Sentry --bust 토글"),
        row("Enter", "큐에 추가"),
        row("Esc", "취소"),
        Line::from(""),
        Line::styled(" [Esc] 닫기", Style::default().fg(Color::DarkGray)).alignment(Alignment::Center),
    ];
    let inner = block.inner(area);
    f.render_widget(Paragraph::new(rows), inner);
}

fn row(key: &str, desc: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {:<16} ", key), Style::default().fg(Color::Yellow)),
        Span::styled(desc.to_string(), Style::default().fg(Color::White)),
    ])
}
