# translate-dashboard Nickel 설정
{
  # 백엔드 번역 API 엔드포인트 (round-robin)
  api_endpoints = [
    "http://10.0.60.108:8080",
    "http://10.0.60.108:8081",
    "http://10.0.60.108:8082",
    "http://10.0.60.108:8083",
  ],

  # GPU 호스트 (nvidia-smi 폴링용 SSH)
  gpu = {
    host = "192.168.2.60",
    gpu_ids = [0, 1, 2, 3],
    poll_interval_s = 5,
  },

  # 번역 기본값
  defaults = {
    source_lang = "en",
    target_lang = "ko",
    workers = 32,
    context = "concise UI text in Korean",
  },

  # Job 타입별 설정
  jobs = {
    translate = {
      cli = "/usr/local/bin/phs-translate",
    },
    sentry_i18n = {
      cli = "/usr/local/bin/phs-sentry-i18n",
      state_path = "/opt/sentry-i18n/state.json",
    },
  },

  # UI
  ui = {
    refresh_ms = 250,
    history_max = 100,
  },

  # 번역 커넥터 (여러 개 등록해두고 런타임에 스위치)
  connectors = {
    default = "gemma",                 # gemma | deepl | claude

    # 사용하려면 api_key 채우고 블록 유지, 안 쓰면 삭제
    # deepl = {
    #   api_key = "xxxx",
    #   pro = false,
    # },
    # claude = {
    #   api_key = "sk-ant-...",
    #   model = "claude-opus-4-6",
    # },
  },
}
