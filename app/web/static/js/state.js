export function detectLang() {
  const fromUrl = new URLSearchParams(location.search).get('lang');
  if (fromUrl === 'en' || fromUrl === 'ko') return fromUrl;
  const fromStorage = localStorage.getItem('lang');
  if (fromStorage === 'en' || fromStorage === 'ko') return fromStorage;
  return 'ko';
}
export const LANG = detectLang();
localStorage.setItem('lang', LANG);
document.documentElement.lang = LANG;

// Theme: explicit choice (localStorage) wins; otherwise CSS's own
// prefers-color-scheme media query decides, so no data-theme attribute is
// set at all until the user actually picks one — applied immediately here
// (before first paint) to avoid a light-then-dark flash on load.
export const _savedTheme = localStorage.getItem('theme');
if (_savedTheme === 'light' || _savedTheme === 'dark') {
  document.documentElement.setAttribute('data-theme', _savedTheme);
}
if (new URLSearchParams(location.search).get('lang') !== LANG) {
  const url = new URL(location.href);
  url.searchParams.set('lang', LANG);
  history.replaceState(null, '', url);
}

// A URL for the *other* language, for the topbar switcher — a real link
// (not a JS-only toggle), so clicking it is a genuine page navigation/
// reload with the new language reflected in the address bar, same pattern
// as any bilingual site's language switcher.
export function langUrl(lang) {
  const url = new URL(location.href);
  url.searchParams.set('lang', lang);
  return url.toString();
}

export const I18N = {
  ko: {
    kisConnected: '● KIS 모의투자 연결됨',
    kisConnectedReal: '🔴 KIS 실전투자 연결됨 — 실제 자금',
    connectionLostBanner: '서버에 연결할 수 없습니다 — 화면의 데이터가 최신이 아닙니다. 연결이 복구되면 자동으로 다시 불러옵니다.',
    execModeManual: '수동',
    execModeSemiAuto: '반자동',
    execModeFullAuto: '완전자동',
    autoTradeScanning: '후보 스캔 중',
    autoTradeAnalyzing: ticker => `분석 중: ${ticker}`,
    autoTradeIdle: mins => `대기 중 · 다음 사이클 ${mins}분 후`,
    autoTradeIdleNoTime: '대기 중',
    autoTradeCardAnalyzing: '분석 중...',
    signalsTitle: '오늘의 AI 시그널',
    signalsSectionHint: '최신순',
    marketWidgetKospi: '코스피',
    marketWidgetKosdaq: '코스닥',
    marketWidgetUsdKrw: '원/달러',
    marketWidgetNasdaq: '나스닥100',
    marketWidgetVix: 'VIX',
    // 랜딩 히어로(#nav-signals 상단) — "이게 뭔지 / 방금 뭘 했는지 / 지금 내가
    // 할 일이 있는지" 세 가지를 답함. 실행모드/계좌모드/장중여부는 헤더에 이미
    // 상시 노출되므로 여기서 중복 표시하지 않음(2026-08-18).
    deskHeroTitle: 'AI 트레이딩 데스크',
    deskHeroTagline: '6개 에이전트가 서로 검증하며 매매를 판단합니다',
    deskChainAnalysts: '분석가 4',
    deskChainDebate: '강세·약세 토론',
    pipelineStageResearch: '리서치 매니저 종합',
    pipelineStageTrader: '트레이더 제안',
    deskChainRisk: '리스크 검토',
    deskChainPm: '최종 판단',
    pipelineStageExecution: '체결·기록',
    deskStatusManual: '수동 모드 · 자동 분석이 꺼져 있습니다',
    deskStatusOffline: '서버에 연결할 수 없습니다 — 분석 상태를 확인할 수 없습니다',
    deskStatusScanning: '후보 종목 스캔 중…',
    deskStatusAnalyzing: n => `지금 ${n}종목 분석 중`,
    deskStatusDone: (time, n) => `${time} 분석 완료 · ${n}종목`,
    deskStatusNeverRan: '아직 첫 사이클 전입니다',
    deskStatusNextCycle: m => `다음 사이클 ${m}분 후`,
    deskPendingCta: n => `승인 대기 ${n}건 — 확인하기`,
    signalConfidenceLabel: '신뢰도',
    signalPriceTargetLabel: '목표가',
    signalTokensUsed: n => `토큰 ${n.toLocaleString('ko-KR')}개 사용`,
    signalAiRatingHint: 'AI 등급',
    signalExecutionNoteLabel: '실행 결과',
    signalManualBadge: '수동 분석',
    signalManualBadgeTitle: '사람이 직접 요청한 분석입니다 — 자동매매 사이클의 판단이 아닙니다',
    signalManualGroupTitle: '수동 분석',
    signalManualGroupHint: '사람이 직접 요청한 판단 · 자동 사이클과 별도 집계',
    signalReanalyzeBtn: '다시 분석',
    signalReanalyzeRunning: '분석 중...',
    signalReanalyzeCooldown: sec => `다시 분석 (${sec}s)`,
    connecting: '연결 중…',
    live: '실시간',
    reconnecting: '재연결 중…',
    heroTotalLabel: '총평가금액',
    heroEvalLabel: '평가금액',
    heroCashLabel: '예수금',
    heroForeignAcctCashLabel: '원화예수금',
    heroInvestedLabel: '매입금액',
    heroD2CashLabel: 'D+2예수금',
    heroForeignCashLabel: '외화예수금',
    heroPositionsLabel: '보유종목',
    pnlLabel: '평가손익',
    sidebarAcctLabel: '내 평가자산 · 국내(KRW)',
    sidebarAcctDetailLink: '상세 ›',
    scopeDomestic: '국내 (KRW)',
    scopeOverseas: '해외 (USD)',
    scopeOverseasKrw: '해외 (KRW)',
    scopeDomesticShort: '국내',
    scopeOverseasShort: '해외',
    themeDark: '다크',
    themeLight: '라이트',
    executionModeLabel: '매매모드',
    sidebarLangLabel: '언어',
    sidebarThemeLabel: '테마',
    sidebarNotifLabel: '알림',
    globalSearchPlaceholder: '종목 검색',
    searchNotFoundTitle: '검색 결과 없음',
    searchNotFoundSub: (q) => `'${q}'에 해당하는 종목을 찾을 수 없습니다.`,
    notifAria: '알림',
    accountModeReal: '실거래',
    accountModePaper: '모의투자',
    fxToggleToKRW: '⇄ KRW', fxToggleToUSD: '⇄ USD', tabSettings: '설정',
    plusPaper: n => ` (+페이퍼 ${n})`,
    collapse: '접기 ▴',
    showMore: n => `더보기 (${n}개) ▾`,
    showAllTradeAllowlist: n => `전체 ${n}종목 보기 ▾`,
    feedTitle: '판단 이력',
    feedEmpty: '아직 활동 기록이 없습니다.',
    todayCount: n => `오늘 ${n}건`,
    itemCount: n => `${n}건`,
    tabTrades: '최근 거래',
    tabPnl: '매매손익',
    pnlStartLabel: '시작일', pnlEndLabel: '종료일', pnlQueryButton: '조회',
    pnlSellDate: '매도일', pnlBuyAvgPrice: '매수평단가', pnlSellPrice: '매도가',
    pnlFeeTax: '수수료/세금', pnlNetPnl: '순손익', pnlNetPnlPct: '순수익률',
    pnlSummaryTotal: '기간 총손익(세전)', pnlSummaryFeeTax: '수수료/세금 합계', pnlSummaryNetTotal: '기간 순손익(세후)',
    pnlSummaryWinRate: '승률',
    pnlEmpty: '조회 기간에 실현된 매매손익이 없습니다.',
    pnlClampedNotice: '조회 시작일이 조회 가능한 최대 기간(약 3개월 전)보다 이전이라 그 시점으로 조정됐습니다 — 3개월보다 오래 보유한 종목은 매수평단가가 부정확할 수 있습니다.',
    pnlFeeUnavailableOverseasNotice: '해외 매매의 수수료/세금은 KIS가 제공하지 않아 측정되지 않습니다 — 해외 실현손익·순손익·승률은 수수료/세금 미차감(총액) 기준입니다.',
    // 적중률(자체 감사) 탭 — /api/audit/accuracy. 승률은 "방향 인식" 채점이라
    // 매수계열은 알파>0, 매도계열은 알파<0이 정답(app/audit/stats.py _WIN_DIRECTION).
    tabAccuracy: '적중률',
    accuracyPanelHint: 'AI 판단의 실현 결과 자체 감사',
    accuracyFilterMarket: '시장', accuracyFilterScope: '집계 대상', accuracyFilterBasis: '수익 기준',
    accuracyMarketAll: '전체',
    accuracyScopeLive: '실거래', accuracyScopeBacktest: '백테스트',
    accuracyBasisVariable: '실보유기간', accuracyBasisHorizon: '10거래일 고정',
    accuracyKpiDirectionalN: '방향성 판단 수', accuracyKpiWinRate: '방향성 승률',
    accuracyKpiUnscoredHold: '미채점(Hold)', accuracyKpiExcluded: '집계 제외',
    accuracyRatingTableTitle: '등급별 적중률',
    accuracyThRating: '등급', accuracyThN: '표본수', accuracyThWinRate: '승률',
    accuracyThMeanAlpha: '평균 알파', accuracyThStdevAlpha: '알파 표준편차', accuracyThMeanRaw: '평균 절대수익',
    accuracyThWinCondition: '정답 조건', accuracyThExcluded: '제외',
    accuracyRatingLabelBuy: 'Buy (매수)', accuracyRatingLabelOverweight: 'Overweight (비중확대)',
    accuracyRatingLabelHoldHeld: 'Hold (보유 유지)', accuracyRatingLabelHoldFlat: 'Hold (현금 유지)',
    accuracyRatingLabelHoldUnknown: 'Hold (보유/현금 미기록)',
    accuracyRatingLabelUnderweight: 'Underweight (비중축소)', accuracyRatingLabelSell: 'Sell (매도)',
    accuracyWinCondUp: '알파 > 0', accuracyWinCondDown: '알파 < 0', accuracyWinCondUnscored: '미채점',
    accuracyAccountModePaper: '모의계좌 기준', accuracyAccountModeReal: '실전계좌 기준',
    accuracyThresholdHint: n => `표본수 ${n}건 미만인 등급의 승률은 회색으로 표시 — 참고용이며 통계적 근거로 쓸 수 없습니다.`,
    accuracyLowNTitle: n => `표본수가 ${n}건 미만이라 승률의 신뢰구간이 매우 넓습니다.`,
    accuracyRatingFootnote: '승률은 방향 인식 채점입니다 — 매수·비중확대는 알파 > 0이 정답, 매도·비중축소는 알파 < 0이 정답(벤치마크가 더 나은 종목에서 빠져나온 것도 옳은 판단). 평균 알파·평균 절대수익은 보유기간 가중이나 연환산이 없는 단순 평균이라 색상 강조를 하지 않습니다. "제외"는 저장된 수익률이 NULL/NaN이어서 통째로 빠진 행 수입니다.',
    accuracyEmpty: '아직 결과가 확정된 판단이 없습니다 — 이 조건으로는 채점할 표본이 없습니다.',
    accuracyLoadError: '적중률 집계를 불러오지 못했습니다 — 서버 로그를 확인하세요.',
    // --- 아래는 Stage 3(마크업만 선반영)에서 쓰는 키 ---
    accuracyConfidenceTableTitle: '확신도 구간별 적중률',
    accuracyThBand: '확신도 구간', accuracyThSizeMultiplier: '사이징 배수', accuracyThMeanSignedAlpha: '평균 부호정렬 알파',
    accuracyWeeklyTableTitle: '주간 추이',
    accuracyThThisWeek: '이번 주', accuracyThLast8Weeks: '최근 8주', accuracyThCumulative: '누적',
    accuracyF7TableTitle: '리스크 토론 오버라이드 결과',
    accuracyThF7Row: '구분', accuracyThWins: '적중',
    accuracyDebateTitle: '리스크 토론 영향',
    accuracyDebateN: '토론 기록 있는 판단', accuracyDebateChanged: '결론이 바뀐 판단',
    accuracySkippedTag: '미집행',
    accuracyCorrNA: '표본 부족으로 계산 불가',
    accuracyConfidenceEmpty: '확신도가 기록된 판단이 아직 없습니다 — 확신도 컬럼(2026-08-13 추가) 이전 판단만 있습니다.',
    accuracyConfidenceFootnote: (nMissing, corr) => `'미집행' 구간(<0.50)은 실제로 집행된 적이 한 번도 없습니다 — app/execution/sizing.py 가 확신도 0.50 미만이면 매매를 아예 건너뛰기 때문에, 이 줄의 승률은 "그래도 집행했다면 어땠을까"를 계산한 반사실(counterfactual) 수치입니다. 사이징 배수는 sizing.py 가 그 확신도에서 실제로 적용하는 값이며, 둘이 어긋나면 한쪽이 낡은 것입니다. 확신도 ↔ 부호정렬 알파 상관: ${corr} — 인과가 아니고 표본이 적으면 무의미하니 조사 단서로만 쓰세요. 확신도 미기록으로 모든 구간에서 제외된 판단: ${nMissing}건.`,
    accuracyWeeklyWindow: (label, start, end, thisN, thisUnscorable, l8N, cumN) => `${label} (${start} ~ ${end}, KST) · 이번 주 ${thisN}건(제외 ${thisUnscorable}건) · 최근 8주 ${l8N}건 · 누적 ${cumN}건`,
    accuracyWeeklyCellNA: 'N/A',
    accuracyWeeklyEmpty: '세 기간 모두 결과가 확정된 판단이 없습니다 — 비교할 것이 없습니다.',
    accuracyWeeklyFootnote: n => `각 칸은 "표본수 · 승률"이며, 승률은 등급별 표와 같은 방향 인식 채점입니다. 기간(이번 주/최근 8주/누적)은 결과 확정일(horizon_eval_date) 기준이고, 표본수 ${n}건 미만인 칸은 건수만 보이고 비율은 감춥니다. 'N/A'는 그 기간에 해당 등급의 확정 판단이 아예 없다는 뜻이며 0건과 구분됩니다.`,
    accuracyWeeklyMarketIgnored: '이 표는 위의 시장 필터를 따르지 않습니다 — /api/audit/weekly 가 market 파라미터를 받지 않아 항상 국내+해외 전체 기준입니다.',
    accuracyF7Empty: '트레이더가 매매를 제안했는데 최종 등급이 Hold로 내려간 확정 판단이 없습니다.',
    accuracyF7Footnote: n => `트레이더는 매매를 제안했으나(trader_action ≠ Hold) 리스크 토론/포트폴리오 매니저를 거쳐 최종 등급이 Hold로 내려간 판단만 모은 표입니다. 주간 창을 적용하지 않은 전체 기간 누적입니다(app/audit/weekly.py _compute_f7_panel — 한 주치로는 표본이 절대 차지 않아 매주 '판정 없음'만 찍히기 때문). 표본 ${n}건 미만인 칸은 승률을 아예 표시하지 않고 건수만 보여줍니다 — 이 표는 "토론이 방향성 판단을 조직적으로 죽이는지"를 잡기 위한 것이라, 몇 건으로 낸 잘못된 판정이 판정 없음보다 나쁩니다.`,
    accuracyF7NoVerdictTitle: n => `표본이 ${n}건 미만이라 승률을 표시하지 않습니다 — 건수만 보여줍니다.`,
    accuracyF7NoVerdictUnknownTitle: '판단 시점의 보유 여부(held_at_decision)가 기록되지 않아 정답 방향을 정할 수 없습니다 — 표본수와 무관하게 판정하지 않습니다.',
    accuracyDebateFootnote: '리스크 토론과 포트폴리오 매니저가 트레이더의 최초 Buy/Hold/Sell 방향을 실제로 바꾼 횟수입니다. 이 수치만은 결과 확정을 기다리지 않습니다 — 방향이 바뀌었는지는 판단 시점에 이미 알 수 있기 때문에, 위 표들과 표본이 다르고 "수익 기준" 필터도 적용되지 않습니다. trader_action 이 기록되기 전의 판단은 "안 바뀜"이 아니라 집계에서 제외됩니다. 아래 목록은 "트레이더 제안 → 최종 방향" 별 건수입니다.',
    tabDecisions: '판단 이력',
    tabPending: '승인 대기',
    tabPortfolio: '포트폴리오 상세',
    tabSettings: '설정',
    tabAlerts: '알림',
    tabChat: 'AI에게 질문',
    thDate: '일자', thSide: '구분', thSymbol: '종목', thQty: '수량',
    thFillPrice: '체결가', thStatus: '상태', thVenue: '경로', thReason: '판단 · 사유',
    thRating: '등급', thAvgPrice: '평단가', thCurPrice: '현재가',
    thPnl: '평가손익', thPnlPct: '수익률', thMarketValue: '평가금액',
    thStopLoss: '손절가', thWeightPct: '비중',
    portfolioTotal: '합계',
    pnlPctHint: '매입가 대비 평가손익률', dayChangeHint: '전일 종가 대비 등락률',
    tradesEmpty: '아직 거래 내역이 없습니다.',
    decisionsEmpty: '아직 판단 기록이 없습니다.',
    pendingOrdersEmpty: '대기 중인 주문이 없습니다.',
    detailModalEmpty: '관심종목에서 종목을 선택하면 여기에 상세 정보가 표시됩니다.',
    pendingRequestPrice: '요청가',
    pendingFillPrice: '체결',
    pendingApprove: '승인',
    pendingReject: '거절',
    pendingApproving: '처리 중…',
    pendingApproved: n => `승인 완료: ${n}`,
    pendingRejected: '거절되었습니다.',
    pendingActionFailed: n => `처리 실패: ${n}`,
    pendingApprovalConfirm: '승인 확인',
    pendingApprovalReject: '승인 거절',
    pendingApprovalLater: '나중에',
    pendingHistoryTitle: '처리 기록',
    pendingHistoryEmpty: '아직 처리된 주문이 없습니다.',
    pendingStatusApproved: '승인됨',
    pendingStatusApproveFailed: '승인 후 실패',
    pendingStatusRejected: '거절됨',
    fullAutoDecided: 'AI 자동매매',
    fullAutoNoTrade: '매매 없음 (Hold 또는 보유량 변화 없음)',
    alertsEmpty: '아직 알림이 없습니다.',
    chatPlaceholder: '예: 삼성전자 지금 얼마야?',
    chatSend: '전송',
    chatModelLabel: 'Muse Invest 4.5',
    chatDisclaimer: 'AI 분석은 투자 참고용이며, 최종 판단과 책임은 본인에게 있습니다.',
    chatSendHint: '↵ 전송',
    chatNewAria: '새 대화 시작',
    chatEmptyTitle: '무엇을 도와드릴까요?',
    chatSuggestRisk: '내 포트폴리오 리스크 진단해줘',
    chatSuggestReason: '오늘 AI가 왜 이 판단을 내렸어?',
    chatSuggestStopLoss: '손절선까지 얼마나 남았어?',
    chatSuggestPerformance: '이번주 성과 어때?',
    stopDistRemaining: pct => `현재가 기준 손절가까지 ${pct}% 남음`,
    stopDistBreached: '현재가가 손절가 아래입니다',
    stopDistBreachedShort: '이탈',
    tradeAllowlistSubFavorites: '관심종목',
    tradeAllowlistSubLive: '실시간',
    tradeAllowlistSubRecent: '최근 본',
    tradeAllowlistSubRecentEmpty: '최근 본 종목이 없습니다.',
    tradeAllowlistAddPlaceholder: '종목명 또는 코드 검색',
    tradeAllowlistAddButton: '+ 추가',
    tickerSearchLoading: '검색 중…',
    tickerSearchEmpty: '일치하는 종목이 없습니다',
    tickerSearchEmptySub: '종목 코드를 직접 입력해 추가할 수도 있습니다.',
    tickerSearchError: '검색을 불러오지 못했습니다',
    tickerSearchRetry: '다시 시도',
    tickerSearchCountAria: (n) => `검색 결과 ${n}건`,
    tickerSearchGroupRecent: '최근 본',
    tickerSearchGroupTopValue: '거래대금 상위',
    tickerSearchBadgeDomestic: '국내',
    tickerSearchBadgeOverseas: '해외',
    tickerSearchBadgeOnList: '관심',
    tickerSearchBlockedLeveraged: '레버리지/인버스 제외',
    tickerSearchAddedTitle: '관심종목에 추가됨',
    tickerSearchAddedSub: (name, code) => `${name} (${code})`,
    tickerSearchAddFailTitle: '추가하지 못했습니다',
    tickerAddErrLeveraged: (name) => `${name}은(는) 레버리지·인버스·선물형 상품이라 이 시스템에서 매매할 수 없습니다.`,
    tickerAddErrUnsupportedType: (name) => `${name}은(는) 지원하지 않는 상품 유형입니다(주식·ETF만 가능).`,
    tickerAddErrUnresolvedName: (code) => `${code}의 종목명을 확인할 수 없어 추가를 거부했습니다.`,
    tickerAddErrCashOrBond: (name) => `${name}은(는) 현금성·채권형 펀드라 분석 대상이 아닙니다(실적·뉴스 없음).`,
    tickerAddErrDerivativeIncome: (name) => `${name}은(는) 커버드콜·옵션인컴형 펀드라 분석 대상이 아닙니다(옵션 기반 수익 구조).`,
    tickerAddErrInvalidFormat: (q) => `'${q}'은(는) 종목 코드 형식이 아닙니다. 목록에서 골라 주세요.`,
    tickerAddErrNotFound: (q) => `'${q}'에 해당하는 종목을 찾을 수 없습니다.`,
    modalClose: '✕ 닫기',
    modalCloseAria: '닫기',
    earningsTrendTitle: '실적 추이',
    periodChartAria: '기간 가격 차트 — 봉 개수와 시가·고가·저가·종가·거래량 수치는 차트 아래에 텍스트로 표시됩니다.',
    epsLabel: '주당순이익 (EPS)',
    revLabel: '매출 vs. 순이익',
    periodQuarterly: '분기',
    periodAnnual: '연간',
    revenueEstimateTitle: '매출 추정치',
    earningsEstimateTitle: '실적 추정치',
    newsTitle: '최근 뉴스',
    waiting: '대기 중…',
    openDetailAria: name => `${name} — 상세 차트 열기`,
    marketOpen: '장중',
    marketClosed: '장마감',
    newsLoading: '뉴스 불러오는 중…',
    newsEmpty: '최근 뉴스가 없습니다.',
    newsError: '뉴스를 불러오지 못했습니다.',
    minutesAgo: n => `${n}분 전`,
    hoursAgo: n => `${n}시간 전`,
    daysAgo: n => `${n}일 전`,
    period0q: '이번 분기', 'period+1q': '다음 분기', period0y: '올해', 'period+1y': '내년',
    estimatesEmpty: '추정치 데이터가 없습니다.',
    estAvg: '평균 추정', estLow: '최저 추정', estHigh: '최고 추정',
    estYearAgo: '전년 동기', estGrowth: '성장률', estAnalysts: '# Analysts',
    epsPending: '발표 예정',
    epsBeat: '상회', epsMiss: '하회',
    estimateLabel: '추정치', actualLabel: '실제',
    revenueLabel: '매출', earningsLabel: '순이익',
    alertPriceKind: '가격', alertVolumeKind: '거래량', alertStopLossKind: '손절 매도',
    alertPipelineFailureKind: '분석 실패', alertCycleFailureKind: '사이클 실패',
    alertOvernightGapKind: '야간 갭 위험',
    alertUnmirroredTradeKind: '체결 미확인',
    alertUnconfirmedFillKind: '체결 결과 이상',
    alertUnconfirmedFillFinalKind: '체결 확인 포기',
    venueKisReal: 'KIS 실전', venueKisPaper: 'KIS 모의', venueKisUnknown: 'KIS', venuePaper: '페이퍼 원장',
    holdingBadge: n => `보유 ${n.toLocaleString('en-US')}주`,
    sharesUnit: '주',
    avgPricePrefix: ' · 평단 ',
    buy: '매수', sell: '매도',
    statusFilled: '체결', statusPartial: '부분체결', statusPending: '미체결', statusRecorded: '기록',
    seeReasoning: '▸ 판단 보기', hideReasoning: '▾ 접기',
    loading: '불러오는 중…',
    transcriptError: '토론 기록을 불러오지 못했습니다.',
    noTranscript: '기록된 토론이 없습니다 — 토론 기록 기능이 추가되기 전의 판단입니다.',
    transcriptAnalysts: '애널리스트 보고서',
    transcriptDebate: '강세 vs. 약세 토론',
    transcriptResearchManager: '리서치 매니저',
    transcriptTrader: '트레이더',
    transcriptRiskDebate: '리스크 토론',
    transcriptThesis: '투자 논리 상세',
    transcriptPMContext: '포트폴리오 매니저가 참고한 컨텍스트',
    transcriptPastContext: '과거 유사 판단',
    transcriptSiblingContext: '같은 날 다른 판단',
    transcriptPositionContext: '보유 현황',
    transcriptVolatilityContext: '변동성 / 사이징',
    transcriptRiskCapContext: '리스크 한도',
    transcriptPortfolioRiskContext: '포트폴리오 리스크',
    transcriptExecutionStatusContext: '실행 가능 상태',
    degradedAnalystsWarning: n => `⚠️ 애널리스트 ${n}명 실패 — 축소된 분석입니다`,
    roundSuffix: n => `${n}라운드`,
    resolvedAlpha: pct => `확정 — 알파 ${pct}%`,
    pendingOutcome: '결과 미확정',
    seeFull: '▸ 판단 전문 보기', hideFull: '▾ 접기',
    seeDebate: '▸ 에이전트 판단 과정', hideDebate: '▾ 토론 접기',
    thinking: '생각 중…',
    chatError: '죄송해요, 답변 중 문제가 발생했습니다.',
    cbWarn: 'CB 주의',
    tabOrder: '주문', tabRank: '순위',
    orderModeChecking: '모드 확인 중…', orderModeOff: 'KIS 미연결',
    orderModeDisabled: '실행 비활성 · 주문 차단됨',
    orderModeDry: '모의 (dry-run) · 실제 전송 안 됨',
    orderModeLive: '실주문 활성 · 실제 전송됨',
    orderFieldTicker: '종목', orderFieldLimitPrice: '지정가', orderFieldShares: '수량',
    orderLimitPricePlaceholder: '주문 단가', orderSharesPlaceholder: '주문 수량',
    orderTypeMarket: '시장가', orderTypeLimit: '지정가',
    orderSubmit: '주문',
    orderSubmitBuy: '외화매수', orderSubmitSell: '현금매도',
    openOrdersTitle: '미체결 주문', openOrdersEmpty: '미체결 주문이 없습니다.',
    orderAvailBuy: (cash, maxQty) => `매수가능: 현금 ${cash} · 최대 ${maxQty}주`,
    orderAvailBuyApprox: (cash, qty) => `매수가능: 현금 ${cash} · 최대 약 ${qty}주`,
    orderAvailBuyCashOnly: cash => `매수가능: 현금 ${cash} (기준가 없어 수량 계산 불가)`,
    orderAvailBuyUnsupported: '매수가능 조회 실패',
    orderAvailSell: qty => `매도가능: ${qty}주`,
    orderAvailError: '가능수량 조회 실패',
    orderQtyInvalid: '수량을 올바르게 입력하세요.',
    orderQtyQuick10: '10%',
    orderQtyQuick25: '25%',
    orderQtyQuick50: '50%',
    orderQtyQuickMax: '최대',
    orderQtyQuickUnavailable: '주문가능 수량을 확인하는 중입니다',
    orderQtyQuickNoCash: '주문가능 현금이 없습니다',
    orderQtyQuickNoHolding: '보유 수량이 없습니다',
    orderEstimateLabel: '예상 주문금액',
    orderEstimateEmpty: '수량 입력',
    orderEstimateUnknown: '참조 가격 없음',
    orderEstimateOver: n => `주문가능 ${n}주 초과`,
    orderLimitPriceRequired: '지정가를 입력하세요.',
    orderConfirmTitle: '주문 확인',
    orderConfirmAmount: amt => `예상 주문금액 ${amt}`,
    orderConfirmAmountUnknown: '예상 주문금액 확인 불가 (참조 가격 없음)',
    orderApprovalThresholdTitle: '승인 필요',
    orderApprovalThresholdConfirm: (amt, thr) => `이 주문(약 ${amt})은 승인 임계금액(${thr}) 이상입니다.\n승인하고 진행할까요?`,
    orderTypeLimitWithPrice: price => `지정가 ${price}`,
    orderRequestFailed: '주문 요청 실패 — 네트워크 연결을 확인하세요.',
    orderStatusSubmitted: '제출됨', orderStatusDryRun: '모의(dry-run)', orderStatusRejected: '거부됨', orderStatusError: '오류',
    orderLedgerNotRecorded: '주문은 KIS에 접수됨 · 내부 원장 기록 실패 — 재주문 금지, 수동 확인 필요',
    openOrderRowLabel: (name, sideLabel, remaining, total) => `${name} · ${sideLabel} ${remaining}/${total}주`,
    reviseBtn: '정정', cancelBtn: '취소',
    orderNotCancelableTitle: '이 세션에서 넣은 주문만 정정/취소할 수 있습니다',
    revisePricePrompt: '새 지정가를 입력하세요:',
    revisePriceInvalid: '올바른 가격을 입력하세요.',
    cancelConfirm: '이 주문을 취소할까요?',
    reviseFailed: msg => `정정 실패: ${msg}`,
    reviseRequestFailed: '정정 요청 실패 — 네트워크 연결을 확인하세요.',
    cancelFailed: msg => `취소 실패: ${msg}`,
    cancelRequestFailed: '취소 요청 실패 — 네트워크 연결을 확인하세요.',
    fullAutoSwitchConfirm: '완전자동 모드로 전환하면 AI가 직접 주문을 제출합니다. 계속할까요?',
    killSwitchTitle: '긴급 정지',
    killSwitchReasonPrompt: '긴급 정지 사유를 입력하세요 (필수):',
    killSwitchEngageConfirm: '모든 매수/매도 주문을 즉시 차단합니다. 계속할까요?',
    killSwitchDisengageConfirm: '긴급 정지를 해제하고 매매를 재개합니다. 계속할까요?',
    dialogConfirm: '확인', dialogCancel: '취소', dialogClose: '닫기',
    rankModeVolume: '거래량', rankModeValue: '거래대금', rankModeUp: '상승률', rankModeDown: '하락률', rankModeSector: '업종',
    rankRefresh: '새로고침', rankVolumeCol: '거래량',
    rankEmpty: '데이터 없음 (KIS 미연결 또는 장외)',
    rkAdd: '+관심', rkInTradeAllowlist: '관심',
    periodChartTitle: '기간 차트',
    periodMinute: '분', periodDay: '일', periodWeek: '주', periodMonth: '월', periodYear: '년',
    candleChartEmpty: '차트 데이터 없음',
    candleChartNoKisApi: 'KIS API 없음 (해외 종목 연봉)',
    candleChartNoKisApiMinute: 'KIS API 없음 (해외 종목 분봉)',
    candleCountLabel: n => `${n}봉`,
    ohlcOpen: '시', ohlcHigh: '고', ohlcLow: '저', ohlcClose: '종', ohlcVolume: '거래량',
    minuteChartEmptyOff: '분봉 데이터 없음 (장외이거나 미수신)',
    minuteChartEmpty: '분봉 데이터 없음',
    minuteCountLabel: n => `${n}분`,
    orderbookTitle: '실시간 호가', orderbookHintDefault: '— 장중에만 수신',
    orderbookHintLive: (price, pct) => `— 현재가 ${price} (${pct})`,
    orderbookEmpty: '호가 데이터 없음 (장 마감 또는 미수신)',
    krInsightsTitle: '한국 투자 데이터',
    krTabInvestorFlow: '수급', krTabFinancials: '재무', krTabOpinion: '투자의견',
    krTabRisk: '리스크', krTabNews: '뉴스',
    krDataEmpty: '데이터 없음', krUnsupported: '한국(KRX) 종목만 제공됩니다.',
    krLoadError: '데이터를 불러오지 못했습니다.', krNewsEmpty: '최근 뉴스 없음',
    krContentDisclaimer: '※ 아래 데이터는 한국어로만 제공됩니다.',
    detailAnalyzeBtn: '🤖 AI 종합분석 실행',
    aiDiagCompactLabel: 'AI 진단',
    fundHigh: '고가', fundLow: '저가', fundVolume: '거래량',
    fundMarketCap: '시가총액', fundPer: 'PER', fundPbr: 'PBR',
    fund52wHigh: '52주 최고', fund52wLow: '52주 최저',
    trillionWonSuffix: '조', multipleSuffix: '배',
    analyzeStarting: '분석 시작…', analyzeThinking: '에이전트들이 분석 중입니다…',
    analyzeRatingLabel: rating => `등급: ${rating}`,
    analyzeErrorNoDecision: ticker => `${ticker} 분석 결과를 낼 수 없습니다.`,
    analyzeErrorFailed: ticker => `${ticker} 분석 중 오류가 발생했습니다.`,
    analyzeErrorGeneric: '분석 중 문제가 발생했습니다.',
    cardRemoveTitle: '관심종목에서 제거',
    removeHeldTitle: '보유 중인 종목 제거',
    removeHeldConfirm: (name, shares) => `${name}은(는) 현재 ${shares}주 보유 중입니다. 목록에서 제거하면 이 종목의 손절/트레일링 스탑 감시가 중단되고, 매도 주문까지 차단됩니다. 계속할까요?`,
    removeHoldingsUnknownConfirm: name => `${name}의 보유 여부를 확인할 수 없습니다(계좌 조회 실패). 실제로 보유 중이라면 제거 즉시 손절 감시가 중단되고 매도 주문도 차단됩니다. 계속할까요?`,
    removeFailed: msg => `제거 실패: ${msg}`,
    removeRequestFailed: '제거 요청 실패 — 네트워크 연결을 확인하세요.',
    marketStatusCb: '🔴 CB(거래중단)', marketStatusHalt: '⛔ 거래정지',
    navSignals: 'AI 시그널', navTrading: '트레이딩', navAssistant: '어시스턴트',
    // 설정 탭 토글(#rt-execution-mode)의 실제 라벨 — 사이드바 읽기전용 텍스트도 재사용.
    // execModeFullAuto("완전자동")는 지우지 말 것: 킬스위치급 확인창 제목으로만
    // 남아있고(trading.js의 setRuntimeSetting), 여기 짧은 라벨과는 다른 용도.
    execModeFullAutoShort: '자동',
    sidebarCollapse: '사이드바 접기', sidebarExpand: '펼치기',
    settingsConnStatusTitle: '연결 상태',
    settingsTradingEnvTitle: '거래 환경 설정',
    settingsTradingEnvHint: '저장 즉시 반영됩니다(재시작 불필요). 거래 환경은 안전상 서버 설정 파일에서만 변경합니다.',
    settingsAiProfileLabel: 'AI 프로필',
    aiProfileProduction: '프로덕션', aiProfileTestAggressive: '테스트(공격적)', aiProfileDebug: '디버그',
    settingsExecModeLabel: '거래 모드',
    settingsAccountModeLabel: '거래 환경',
    settingsChecking: '조회 중…',
    settingsRiskRulesTitle: '리스크 규칙',
    settingsRiskRulesHint: '저장 즉시 주문 게이트에 반영됩니다. 마스터 스위치·dry-run은 안전상 서버 설정에서만 변경합니다.',
    riskMaxOrderKrw: '주문금액 상한 (원)', riskMaxOrderUsd: '주문금액 상한 ($)',
    riskPctEnabled: '종목당 최대 비중 제한 사용', riskMaxPct: '최대 비중 (%)',
    riskApprovalEnabled: '승인 임계금액 사용', riskApprovalKrw: '승인 임계 (원)', riskApprovalUsd: '승인 임계 ($)',
    riskSaveBtn: '저장',
    settingsSafetyTitle: '안전장치',
    settingsSafetyHint: '킬스위치는 매수/매도 전부를 즉시 차단하는 비상정지입니다. 일일 손실 서킷브레이커는 오늘 시작 자산 대비 손실이 한도를 넘으면 신규 매수만 자동 차단합니다(매도는 항상 가능).',
    safetyKillSwitchLabel: '킬스위치',
    safetyDailyPnlDomestic: '오늘 손익 (국내, KST 00:00 기준)',
    safetyDailyPnlOverseas: '오늘 손익 (해외, 미국 동부시간 00:00 기준)',
    runtimeChangeFailed: msg => `변경 실패: ${msg}`,
    runtimeChangeSaved: '변경됨 · 즉시 반영됩니다.',
    runtimeChangeRequestFailed: '변경 요청 실패 — 네트워크 연결을 확인하세요.',
    riskCapZero: (label, cap) => `${label} 주문 한도가 ${cap}입니다 — 0이면 모든 주문이 차단됩니다. 0보다 큰 금액을 입력하세요.`,
    riskThresholdTooHigh: (label, thr, cap) => `${label} 승인 임계금액(${thr})은 주문 한도(${cap})보다 작아야 합니다. 한도를 넘는 주문은 승인과 무관하게 차단되므로, 같거나 크면 임계값이 무의미해집니다.`,
    riskSaveFailed: msg => `저장 실패: ${msg}`,
    riskSaved: '저장됨 · 주문 게이트에 반영됩니다.',
    riskSaveRequestFailed: '저장 요청 실패 — 네트워크 연결을 확인하세요.',
    dailyLossUnavailable: '정보 없음',
    dailyLossBadge: (pnl, thr) => `${pnl}% (손실 한도 ${thr}%)`,
    dailyLossBreachedSuffix: ' — 신규 매수 차단됨',
    safetyLoadFailed: '조회 실패',
    killSwitchEngagedBadge: reason => `정지됨 (${reason})`,
    killSwitchNoReason: '사유 없음',
    killSwitchOkBadge: '정상 (매매 가능)',
    killSwitchDisengageBtn: '정지 해제',
    killSwitchActionFailed: msg => `실패: ${msg}`,
    killSwitchEngagedResult: '긴급 정지되었습니다.',
    killSwitchDisengagedResult: '정지가 해제되었습니다.',
    killSwitchRequestFailed: '요청 실패 — 네트워크 연결을 확인하세요.',
  },
  en: {
    kisConnected: '● KIS paper trading connected',
    kisConnectedReal: '🔴 KIS REAL trading connected — real money',
    connectionLostBanner: 'Cannot reach the server — the data on screen is not current. It will reload automatically once the connection is back.',
    execModeManual: 'Manual',
    execModeSemiAuto: 'Semi-Auto',
    execModeFullAuto: 'Full-Auto',
    autoTradeScanning: 'Scanning candidates',
    autoTradeAnalyzing: ticker => `Analyzing: ${ticker}`,
    autoTradeIdle: mins => `Idle · next cycle in ${mins}m`,
    autoTradeIdleNoTime: 'Idle',
    autoTradeCardAnalyzing: 'Analyzing...',
    signalsTitle: "Today's AI Signals",
    signalsSectionHint: 'Most recent first',
    marketWidgetKospi: 'KOSPI',
    marketWidgetKosdaq: 'KOSDAQ',
    marketWidgetUsdKrw: 'USD/KRW',
    marketWidgetNasdaq: 'Nasdaq 100',
    marketWidgetVix: 'VIX',
    deskHeroTitle: 'AI Trading Desk',
    deskHeroTagline: 'Six agents cross-check each other before any trade decision',
    deskChainAnalysts: '4 analysts',
    deskChainDebate: 'Bull vs. bear debate',
    pipelineStageResearch: 'Research synthesis',
    pipelineStageTrader: 'Trader proposal',
    deskChainRisk: 'Risk review',
    deskChainPm: 'Final call',
    pipelineStageExecution: 'Execution',
    deskStatusManual: 'Manual mode · automatic analysis is off',
    deskStatusOffline: 'Cannot reach the server — analysis status unavailable',
    deskStatusScanning: 'Scanning for candidates…',
    deskStatusAnalyzing: n => `Analyzing ${n} ticker(s) now`,
    deskStatusDone: (time, n) => `Last run ${time} · ${n} tickers`,
    deskStatusNeverRan: 'The first cycle has not run yet',
    deskStatusNextCycle: m => `Next cycle in ${m} min`,
    deskPendingCta: n => `${n} order(s) awaiting your approval — review`,
    signalConfidenceLabel: 'Confidence',
    signalPriceTargetLabel: 'Target',
    signalTokensUsed: n => `${n.toLocaleString('en-US')} tokens used`,
    signalAiRatingHint: 'AI rating',
    signalExecutionNoteLabel: 'Execution',
    signalManualBadge: 'Manual',
    signalManualBadgeTitle: 'Requested by a human — not a scheduled auto-trade judgment',
    signalManualGroupTitle: 'Manual analyses',
    signalManualGroupHint: 'Human-requested runs · counted separately from the auto cycle',
    signalReanalyzeBtn: 'Re-analyze',
    signalReanalyzeRunning: 'Analyzing...',
    signalReanalyzeCooldown: sec => `Re-analyze (${sec}s)`,
    connecting: 'Connecting…',
    live: 'Live',
    reconnecting: 'Reconnecting…',
    heroTotalLabel: 'Total value',
    heroEvalLabel: 'Eval. amount',
    heroCashLabel: 'Cash',
    heroForeignAcctCashLabel: 'KRW cash',
    heroInvestedLabel: 'Invested amount',
    heroD2CashLabel: 'D+2 cash',
    heroForeignCashLabel: 'Foreign currency cash',
    heroPositionsLabel: 'Positions',
    pnlLabel: 'P&L',
    sidebarAcctLabel: 'My Equity · Domestic (KRW)',
    sidebarAcctDetailLink: 'Details ›',
    scopeDomestic: 'Domestic (KRW)',
    scopeOverseas: 'Overseas (USD)',
    scopeOverseasKrw: 'Overseas (KRW)',
    scopeDomesticShort: 'Domestic',
    scopeOverseasShort: 'Overseas',
    themeDark: 'Dark',
    themeLight: 'Light',
    executionModeLabel: 'Mode',
    sidebarLangLabel: 'Language',
    sidebarThemeLabel: 'Theme',
    sidebarNotifLabel: 'Notifications',
    globalSearchPlaceholder: 'Search ticker',
    searchNotFoundTitle: 'No matches',
    searchNotFoundSub: (q) => `No ticker found for '${q}'.`,
    notifAria: 'Notifications',
    accountModeReal: 'Live',
    accountModePaper: 'Paper',
    fxToggleToKRW: '⇄ KRW', fxToggleToUSD: '⇄ USD', tabSettings: 'Settings',
    plusPaper: n => ` (+${n} paper)`,
    collapse: 'Collapse ▴',
    showMore: n => `Show ${n} more ▾`,
    showAllTradeAllowlist: n => `Show all ${n} ▾`,
    feedTitle: 'Decision History',
    feedEmpty: 'No activity yet.',
    todayCount: n => `${n} today`,
    itemCount: n => `${n}`,
    tabTrades: 'Recent Trades',
    tabPnl: 'Realized P&L',
    pnlStartLabel: 'Start', pnlEndLabel: 'End', pnlQueryButton: 'Query',
    pnlSellDate: 'Sell Date', pnlBuyAvgPrice: 'Buy Avg Price', pnlSellPrice: 'Sell Price',
    pnlFeeTax: 'Fee/Tax', pnlNetPnl: 'Net P&L', pnlNetPnlPct: 'Net Return',
    pnlSummaryTotal: 'Total P&L (gross)', pnlSummaryFeeTax: 'Total Fee/Tax', pnlSummaryNetTotal: 'Total P&L (net)',
    pnlSummaryWinRate: 'Win Rate',
    pnlEmpty: 'No realized P&L in this date range.',
    pnlClampedNotice: 'The start date was adjusted forward to the earliest date this system can query (about 3 months back) — a position held longer than that may show an inaccurate average buy price.',
    pnlFeeUnavailableOverseasNotice: 'Overseas commission/tax is not exposed by KIS and is not measured — overseas realized P&L, net P&L, and win rate below are gross (before fees/tax).',
    tabAccuracy: 'Hit Rate',
    accuracyPanelHint: 'Self-audit of realized AI decision outcomes',
    accuracyFilterMarket: 'Market', accuracyFilterScope: 'Population', accuracyFilterBasis: 'Return basis',
    accuracyMarketAll: 'All',
    accuracyScopeLive: 'Live', accuracyScopeBacktest: 'Backtest',
    accuracyBasisVariable: 'Actual holding period', accuracyBasisHorizon: 'Fixed 10 trading days',
    accuracyKpiDirectionalN: 'Directional decisions', accuracyKpiWinRate: 'Directional win rate',
    accuracyKpiUnscoredHold: 'Unscored (Hold)', accuracyKpiExcluded: 'Excluded',
    accuracyRatingTableTitle: 'Hit rate by rating',
    accuracyThRating: 'Rating', accuracyThN: 'N', accuracyThWinRate: 'Win rate',
    accuracyThMeanAlpha: 'Mean alpha', accuracyThStdevAlpha: 'Alpha σ', accuracyThMeanRaw: 'Mean raw',
    accuracyThWinCondition: 'Win condition', accuracyThExcluded: 'Excluded',
    accuracyRatingLabelBuy: 'Buy', accuracyRatingLabelOverweight: 'Overweight',
    accuracyRatingLabelHoldHeld: 'Hold (held)', accuracyRatingLabelHoldFlat: 'Hold (flat)',
    accuracyRatingLabelHoldUnknown: 'Hold (held/flat unknown)',
    accuracyRatingLabelUnderweight: 'Underweight', accuracyRatingLabelSell: 'Sell',
    accuracyWinCondUp: 'alpha > 0', accuracyWinCondDown: 'alpha < 0', accuracyWinCondUnscored: 'unscored',
    accuracyAccountModePaper: 'Paper account', accuracyAccountModeReal: 'Real account',
    accuracyThresholdHint: n => `Win rates from fewer than ${n} samples are greyed out — indicative only, not a statistical basis.`,
    accuracyLowNTitle: n => `Fewer than ${n} samples — this win rate has a very wide confidence interval.`,
    accuracyRatingFootnote: 'Win rate is DIRECTION-AWARE: Buy/Overweight win on alpha > 0, Sell/Underweight win on alpha < 0 (stepping out of a name the benchmark then beat is a correct call). Mean alpha and mean raw are simple averages — not annualized, not weighted by holding period — so they are deliberately not colour-coded. "Excluded" counts resolved rows dropped entirely because the stored return was NULL or NaN.',
    accuracyEmpty: 'No resolved decisions yet — nothing to score under these filters.',
    accuracyLoadError: 'Could not load the hit-rate report — check the server log.',
    // --- Stage 3 keys (markup pre-placed, renderers not written yet) ---
    accuracyConfidenceTableTitle: 'Hit rate by confidence band',
    accuracyThBand: 'Band', accuracyThSizeMultiplier: 'Size multiplier', accuracyThMeanSignedAlpha: 'Mean signed alpha',
    accuracyWeeklyTableTitle: 'Weekly trend',
    accuracyThThisWeek: 'This week', accuracyThLast8Weeks: 'Last 8 weeks', accuracyThCumulative: 'Cumulative',
    accuracyF7TableTitle: 'Risk-debate override outcomes',
    accuracyThF7Row: 'Group', accuracyThWins: 'Wins',
    accuracyDebateTitle: 'Risk-debate impact',
    accuracyDebateN: 'Decisions with a debate record', accuracyDebateChanged: 'Decisions the debate changed',
    accuracySkippedTag: 'not executed',
    accuracyCorrNA: 'N/A (too few scorable rows)',
    accuracyConfidenceEmpty: 'No decisions with a recorded confidence yet — every resolved row predates the confidence column (added 2026-08-13).',
    accuracyConfidenceFootnote: (nMissing, corr) => `The '<0.50' band was NEVER EXECUTED — app/execution/sizing.py skips the trade entirely below 0.50, so this row's win rate is a counterfactual: what would have happened had it been executed anyway. The size multiplier is what sizing.py actually applies at that confidence; if the two ever disagree, one of them is stale. Confidence ↔ signed-alpha correlation: ${corr} — correlation is not causation and is meaningless on a small sample, so treat it as a hint to investigate. Decisions excluded from every band for having no recorded confidence: ${nMissing}.`,
    accuracyWeeklyWindow: (label, start, end, thisN, thisUnscorable, l8N, cumN) => `${label} (${start} ~ ${end}, KST) · This week N=${thisN} (excluded ${thisUnscorable}) · Last 8 weeks N=${l8N} · Cumulative N=${cumN}`,
    accuracyWeeklyCellNA: 'N/A',
    accuracyWeeklyEmpty: 'No resolved decisions in any window yet — nothing to compare.',
    accuracyWeeklyFootnote: n => `Each cell reads "N · win rate", scored direction-aware exactly as in the by-rating table. Windows (this week / last 8 weeks / cumulative) are anchored on the resolution date (horizon_eval_date). Cells with fewer than ${n} samples show the count only and hide the percentage. 'N/A' means that window has no resolved decision of that rating at all — deliberately distinct from a count of 0.`,
    accuracyWeeklyMarketIgnored: 'This table does NOT follow the market filter above — /api/audit/weekly takes no market parameter, so it is always domestic + overseas combined.',
    accuracyF7Empty: 'No resolved decisions where the Trader proposed a trade but the final rating came out Hold.',
    accuracyF7Footnote: n => `Decisions where the Trader proposed a trade (trader_action ≠ Hold) but the risk debate / Portfolio Manager brought the final rating down to Hold. Cumulative over all history, with no weekly window (app/audit/weekly.py _compute_f7_panel — a single week never has enough rows, so a windowed version would print "no verdict" forever). Cells with fewer than ${n} samples show NO win rate at all, only the count: this table exists to catch a systematic override problem, and a false read from a handful of rows is worse than none.`,
    accuracyF7NoVerdictTitle: n => `Fewer than ${n} samples — no win rate is shown, only the count.`,
    accuracyF7NoVerdictUnknownTitle: 'held_at_decision was not recorded, so the win direction is not determinable — this row never gets a verdict regardless of sample size.',
    accuracyDebateFootnote: 'How often the risk debate and Portfolio Manager actually changed the Trader\'s initial Buy/Hold/Sell direction. This figure alone does not wait for outcomes to resolve — whether the direction changed is knowable the moment the decision is made — so its sample differs from the tables above and the "Return basis" filter does not apply to it. Decisions written before trader_action existed are excluded, not counted as "unchanged". The list below is the count per "Trader proposal → final direction".',
    tabDecisions: 'Decision History',
    tabPending: 'Pending Approval',
    tabPortfolio: 'Portfolio Detail',
    tabSettings: 'Settings',
    tabAlerts: 'Alerts',
    tabChat: 'Ask AI',
    thDate: 'Date', thSide: 'Side', thSymbol: 'Symbol', thQty: 'Qty',
    thFillPrice: 'Fill Price', thStatus: 'Status', thVenue: 'Venue', thReason: 'Decision · Reason',
    thRating: 'Rating', thAvgPrice: 'Avg Price', thCurPrice: 'Current Price',
    thPnl: 'P&L', thPnlPct: 'Return', thMarketValue: 'Market Value',
    thStopLoss: 'Stop Loss', thWeightPct: 'Weight',
    portfolioTotal: 'Total',
    pnlPctHint: 'Unrealized P&L % vs. average cost', dayChangeHint: "Change % vs. previous close",
    tradesEmpty: 'No trades yet.',
    decisionsEmpty: 'No decisions yet.',
    pendingOrdersEmpty: 'No orders awaiting approval.',
    detailModalEmpty: 'Select a ticker from your watchlist to see its details here.',
    pendingRequestPrice: 'requested',
    pendingFillPrice: 'filled',
    pendingApprove: 'Approve',
    pendingReject: 'Reject',
    pendingApproving: 'Processing…',
    pendingApproved: n => `Approved: ${n}`,
    pendingRejected: 'Rejected.',
    pendingActionFailed: n => `Failed: ${n}`,
    pendingApprovalConfirm: 'Confirm Approval',
    pendingApprovalReject: 'Reject',
    pendingApprovalLater: 'Later',
    pendingHistoryTitle: 'Resolution History',
    pendingHistoryEmpty: 'No resolved orders yet.',
    pendingStatusApproved: 'Approved',
    pendingStatusApproveFailed: 'Approved, then failed',
    pendingStatusRejected: 'Rejected',
    fullAutoDecided: 'AI auto-trade',
    fullAutoNoTrade: 'No trade (Hold or no position change)',
    alertsEmpty: 'No alerts yet.',
    chatPlaceholder: 'e.g. What’s Apple’s price right now?',
    chatSend: 'Send',
    chatModelLabel: 'Muse Invest 4.5',
    chatDisclaimer: 'AI analysis is for reference only — final decisions and responsibility are yours.',
    chatSendHint: '↵ Send',
    chatNewAria: 'Start new chat',
    chatEmptyTitle: 'What can I help with?',
    chatSuggestRisk: 'Diagnose my portfolio risk',
    chatSuggestReason: 'Why did the AI make today’s call?',
    chatSuggestStopLoss: 'How far to my stop-loss?',
    chatSuggestPerformance: 'How’s this week’s performance?',
    stopDistRemaining: pct => `${pct}% above the stop-loss at the current price`,
    stopDistBreached: 'Current price is below the stop-loss',
    stopDistBreachedShort: 'breached',
    tradeAllowlistSubFavorites: 'Favorites',
    tradeAllowlistSubLive: 'Live',
    tradeAllowlistSubRecent: 'Recent',
    tradeAllowlistSubRecentEmpty: 'No recently-viewed tickers yet.',
    tradeAllowlistAddPlaceholder: 'Search by name or ticker',
    tradeAllowlistAddButton: '+ Add',
    tickerSearchLoading: 'Searching…',
    tickerSearchEmpty: 'No matching ticker',
    tickerSearchEmptySub: 'You can also type a ticker code directly.',
    tickerSearchError: 'Search failed to load',
    tickerSearchRetry: 'Retry',
    tickerSearchCountAria: (n) => `${n} results`,
    tickerSearchGroupRecent: 'Recently viewed',
    tickerSearchGroupTopValue: 'Most traded',
    tickerSearchBadgeDomestic: 'KR',
    tickerSearchBadgeOverseas: 'US',
    tickerSearchBadgeOnList: 'Added',
    tickerSearchBlockedLeveraged: 'Leveraged/inverse — not tradable',
    tickerSearchAddedTitle: 'Added to favorites',
    tickerSearchAddedSub: (name, code) => `${name} (${code})`,
    tickerSearchAddFailTitle: 'Could not add',
    tickerAddErrLeveraged: (name) => `${name} is a leveraged/inverse/futures product and cannot be traded by this system.`,
    tickerAddErrUnsupportedType: (name) => `${name} is an unsupported product type (stocks and ETFs only).`,
    tickerAddErrUnresolvedName: (code) => `Could not resolve a company name for ${code} — refusing to add it.`,
    tickerAddErrCashOrBond: (name) => `${name} is a cash-like/bond fund — outside this system's company-analysis scope.`,
    tickerAddErrDerivativeIncome: (name) => `${name} is a covered-call / option-income fund — outside this system's company-analysis scope.`,
    tickerAddErrInvalidFormat: (q) => `'${q}' is not a valid ticker format. Pick one from the list.`,
    tickerAddErrNotFound: (q) => `No ticker found for '${q}'.`,
    modalClose: '✕ Close',
    modalCloseAria: 'Close',
    earningsTrendTitle: 'Earnings Trends',
    periodChartAria: 'Price chart — the bar count and open, high, low, close and volume figures are shown as text below the chart.',
    epsLabel: 'Earnings Per Share (EPS)',
    revLabel: 'Revenue vs. Earnings',
    periodQuarterly: 'Quarterly',
    periodAnnual: 'Annual',
    revenueEstimateTitle: 'Revenue Estimate',
    earningsEstimateTitle: 'Earnings Estimate',
    newsTitle: 'Recent News',
    waiting: 'Waiting…',
    openDetailAria: name => `${name} — open detail chart`,
    marketOpen: 'Open',
    marketClosed: 'Closed',
    newsLoading: 'Loading news…',
    newsEmpty: 'No recent news.',
    newsError: 'Could not load news.',
    minutesAgo: n => `${n}m ago`,
    hoursAgo: n => `${n}h ago`,
    daysAgo: n => `${n}d ago`,
    period0q: 'This quarter', 'period+1q': 'Next quarter', period0y: 'This year', 'period+1y': 'Next year',
    estimatesEmpty: 'No estimate data available.',
    estAvg: 'Avg estimate', estLow: 'Low estimate', estHigh: 'High estimate',
    estYearAgo: 'Year ago', estGrowth: 'Growth', estAnalysts: '# Analysts',
    epsPending: 'Upcoming',
    epsBeat: 'Beat', epsMiss: 'Miss',
    estimateLabel: 'Estimate', actualLabel: 'Actual',
    revenueLabel: 'Revenue', earningsLabel: 'Earnings',
    alertPriceKind: 'Price', alertVolumeKind: 'Volume', alertStopLossKind: 'Stop-loss sell',
    alertPipelineFailureKind: 'Analysis failed', alertCycleFailureKind: 'Cycle failed',
    alertOvernightGapKind: 'Overnight gap risk',
    alertUnmirroredTradeKind: 'Fill unconfirmed',
    alertUnconfirmedFillKind: 'Fill outcome issue',
    alertUnconfirmedFillFinalKind: 'Fill confirmation abandoned',
    venueKisReal: 'KIS real', venueKisPaper: 'KIS paper', venueKisUnknown: 'KIS', venuePaper: 'Paper ledger',
    holdingBadge: n => `Holding ${n.toLocaleString('en-US')} sh.`,
    sharesUnit: ' sh.',
    avgPricePrefix: ' · avg ',
    buy: 'Buy', sell: 'Sell',
    statusFilled: 'Filled', statusPartial: 'Partial', statusPending: 'Pending', statusRecorded: 'Recorded',
    seeReasoning: '▸ See decision', hideReasoning: '▾ Hide',
    loading: 'Loading…',
    transcriptError: 'Could not load the debate transcript.',
    noTranscript: 'No transcript recorded — this decision predates the transcript feature.',
    transcriptAnalysts: 'Analyst Reports',
    transcriptDebate: 'Bull vs. Bear Debate',
    transcriptResearchManager: 'Research Manager',
    transcriptTrader: 'Trader',
    transcriptRiskDebate: 'Risk Debate',
    transcriptThesis: 'Full Investment Thesis',
    transcriptPMContext: 'Context the Portfolio Manager Saw',
    transcriptPastContext: 'Past Similar Decisions',
    transcriptSiblingContext: 'Same-Day Sibling Decisions',
    transcriptPositionContext: 'Position Status',
    transcriptVolatilityContext: 'Volatility / Sizing',
    transcriptRiskCapContext: 'Risk Cap',
    transcriptPortfolioRiskContext: 'Portfolio Risk',
    transcriptExecutionStatusContext: 'Execution Status',
    degradedAnalystsWarning: n => `⚠️ ${n} analyst(s) failed for this decision — degraded analysis`,
    roundSuffix: n => `Round ${n}`,
    resolvedAlpha: pct => `Resolved — alpha ${pct}%`,
    pendingOutcome: 'Result not yet known',
    seeFull: '▸ See full reasoning', hideFull: '▾ Hide',
    seeDebate: '▸ See how the agents debated this', hideDebate: '▾ Hide the debate',
    thinking: 'Thinking…',
    chatError: 'Sorry, something went wrong answering that.',
    cbWarn: 'CB warning',
    tabOrder: 'Order', tabRank: 'Screener',
    orderModeChecking: 'Checking mode…', orderModeOff: 'KIS not connected',
    orderModeDisabled: 'Execution disabled · orders blocked',
    orderModeDry: 'Dry-run · nothing actually sent',
    orderModeLive: 'Live · orders actually sent',
    orderFieldTicker: 'Ticker', orderFieldLimitPrice: 'Limit price', orderFieldShares: 'Shares',
    orderLimitPricePlaceholder: 'Order price', orderSharesPlaceholder: 'Order quantity',
    orderTypeMarket: 'Market', orderTypeLimit: 'Limit',
    orderSubmit: 'Submit order',
    orderSubmitBuy: 'Buy', orderSubmitSell: 'Sell',
    openOrdersTitle: 'Open Orders', openOrdersEmpty: 'No open orders.',
    orderAvailBuy: (cash, maxQty) => `Buying power: ${cash} · max ${maxQty} sh.`,
    orderAvailBuyApprox: (cash, qty) => `Buying power: ${cash} · about ${qty} sh. max`,
    orderAvailBuyCashOnly: cash => `Buying power: ${cash} (no reference price for quantity)`,
    orderAvailBuyUnsupported: 'Buying-power lookup unavailable',
    orderAvailSell: qty => `Sellable: ${qty} sh.`,
    orderAvailError: 'Could not check available quantity',
    orderQtyInvalid: 'Enter a valid quantity.',
    orderQtyQuick10: '10%',
    orderQtyQuick25: '25%',
    orderQtyQuick50: '50%',
    orderQtyQuickMax: 'Max',
    orderQtyQuickUnavailable: 'Checking available quantity…',
    orderQtyQuickNoCash: 'No buying power available',
    orderQtyQuickNoHolding: 'No shares held',
    orderEstimateLabel: 'Est. order amount',
    orderEstimateEmpty: 'Enter quantity',
    orderEstimateUnknown: 'No reference price',
    orderEstimateOver: n => `Exceeds ${n} sh. available`,
    orderLimitPriceRequired: 'Enter a limit price.',
    orderConfirmTitle: 'Confirm Order',
    orderConfirmAmount: amt => `Estimated order amount ${amt}`,
    orderConfirmAmountUnknown: 'Estimated order amount unavailable (no reference price)',
    orderApprovalThresholdTitle: 'Approval Required',
    orderApprovalThresholdConfirm: (amt, thr) => `This order (about ${amt}) is at or above the approval threshold (${thr}).\nApprove and proceed?`,
    orderTypeLimitWithPrice: price => `Limit @ ${price}`,
    orderRequestFailed: 'Order request failed — check your connection.',
    orderStatusSubmitted: 'Submitted', orderStatusDryRun: 'Dry-run', orderStatusRejected: 'Rejected', orderStatusError: 'Error',
    orderLedgerNotRecorded: 'Order reached KIS · internal ledger write failed — do NOT resubmit, verify manually',
    openOrderRowLabel: (name, sideLabel, remaining, total) => `${name} · ${sideLabel} ${remaining}/${total} sh.`,
    reviseBtn: 'Revise', cancelBtn: 'Cancel',
    orderNotCancelableTitle: 'Only orders placed in this session can be revised or cancelled',
    revisePricePrompt: 'Enter the new limit price:',
    revisePriceInvalid: 'Enter a valid price.',
    cancelConfirm: 'Cancel this order?',
    reviseFailed: msg => `Revise failed: ${msg}`,
    reviseRequestFailed: 'Revise request failed — check your connection.',
    cancelFailed: msg => `Cancel failed: ${msg}`,
    cancelRequestFailed: 'Cancel request failed — check your connection.',
    fullAutoSwitchConfirm: 'Switching to Full-Auto mode lets the AI submit orders on its own. Continue?',
    killSwitchTitle: 'Emergency Stop',
    killSwitchReasonPrompt: 'Enter the reason for the emergency stop (required):',
    killSwitchEngageConfirm: 'This immediately blocks all buy/sell orders. Continue?',
    killSwitchDisengageConfirm: 'This releases the emergency stop and resumes trading. Continue?',
    dialogConfirm: 'OK', dialogCancel: 'Cancel', dialogClose: 'Close',
    rankModeVolume: 'Volume', rankModeValue: 'Value', rankModeUp: 'Gainers', rankModeDown: 'Losers', rankModeSector: 'Sectors',
    rankRefresh: 'Refresh', rankVolumeCol: 'Volume',
    rankEmpty: 'No data (KIS not connected, or market closed)',
    rkAdd: '+ Add', rkInTradeAllowlist: 'Added',
    periodChartTitle: 'Price Chart',
    periodMinute: 'Min', periodDay: 'D', periodWeek: 'W', periodMonth: 'M', periodYear: 'Y',
    candleChartEmpty: 'No chart data',
    candleChartNoKisApi: 'No KIS API (yearly candles, overseas tickers)',
    candleChartNoKisApiMinute: 'No KIS API (minute candles, overseas tickers)',
    candleCountLabel: n => `${n} bars`,
    ohlcOpen: 'O', ohlcHigh: 'H', ohlcLow: 'L', ohlcClose: 'C', ohlcVolume: 'Vol',
    minuteChartEmptyOff: 'No minute data (market closed or not yet received)',
    minuteChartEmpty: 'No minute data',
    minuteCountLabel: n => `${n} min`,
    orderbookTitle: 'Live Order Book', orderbookHintDefault: '— received during market hours only',
    orderbookHintLive: (price, pct) => `— last ${price} (${pct})`,
    orderbookEmpty: 'No order book data (market closed or not yet received)',
    krInsightsTitle: 'Korean Market Data',
    krTabInvestorFlow: 'Flows', krTabFinancials: 'Financials', krTabOpinion: 'Opinions',
    krTabRisk: 'Risk', krTabNews: 'News',
    krDataEmpty: 'No data', krUnsupported: 'Available for Korean (KRX) tickers only.',
    krLoadError: 'Could not load data.', krNewsEmpty: 'No recent news.',
    krContentDisclaimer: '※ The data below is provided in Korean only.',
    detailAnalyzeBtn: '🤖 Run AI Analysis',
    aiDiagCompactLabel: 'AI Read',
    fundHigh: 'High', fundLow: 'Low', fundVolume: 'Volume',
    fundMarketCap: 'Market Cap', fundPer: 'P/E', fundPbr: 'P/B',
    fund52wHigh: '52W High', fund52wLow: '52W Low',
    trillionWonSuffix: '조', multipleSuffix: 'x',
    analyzeStarting: 'Starting analysis…', analyzeThinking: 'Agents are analyzing…',
    analyzeRatingLabel: rating => `Rating: ${rating}`,
    analyzeErrorNoDecision: ticker => `Could not produce an analysis result for ${ticker}.`,
    analyzeErrorFailed: ticker => `An error occurred while analyzing ${ticker}.`,
    analyzeErrorGeneric: 'Something went wrong during analysis.',
    cardRemoveTitle: 'Remove from list',
    removeHeldTitle: 'Remove a held position',
    removeHeldConfirm: (name, shares) => `${name} still has ${shares} share(s) held. Removing it stops stop-loss / trailing-stop monitoring for this ticker and blocks sell orders too. Continue?`,
    removeHoldingsUnknownConfirm: name => `Could not confirm whether ${name} is currently held (account lookup failed). If it is held, removing it immediately stops stop-loss monitoring and blocks sell orders. Continue?`,
    removeFailed: msg => `Remove failed: ${msg}`,
    removeRequestFailed: 'Remove request failed — check your network connection.',
    marketStatusCb: '🔴 CB (halted)', marketStatusHalt: '⛔ Trading halted',
    navSignals: 'AI Signals', navTrading: 'Trading', navAssistant: 'Assistant',
    // The settings-tab toggle's (#rt-execution-mode) actual label — also
    // reused by the sidebar's read-only text. Keep execModeFullAuto ("Full-Auto") too:
    // it survives as the kill-switch-grade confirm-dialog title only
    // (trading.js's setRuntimeSetting), a different purpose from this
    // short control label.
    execModeFullAutoShort: 'Auto',
    sidebarCollapse: 'Collapse sidebar', sidebarExpand: 'Expand',
    settingsConnStatusTitle: 'Connection Status',
    settingsTradingEnvTitle: 'Trading Environment',
    settingsTradingEnvHint: 'Changes take effect immediately (no restart). The trading environment itself can only be changed in the server config file, for safety.',
    settingsAiProfileLabel: 'AI Profile',
    aiProfileProduction: 'Production', aiProfileTestAggressive: 'Test (Aggressive)', aiProfileDebug: 'Debug',
    settingsExecModeLabel: 'Trading Mode',
    settingsAccountModeLabel: 'Account Environment',
    settingsChecking: 'Checking…',
    settingsRiskRulesTitle: 'Risk Rules',
    settingsRiskRulesHint: 'Saved changes apply to the order gate immediately. The master switch and dry-run can only be changed in the server config, for safety.',
    riskMaxOrderKrw: 'Max Order Amount (KRW)', riskMaxOrderUsd: 'Max Order Amount (USD)',
    riskPctEnabled: 'Enable max weight per position', riskMaxPct: 'Max Weight (%)',
    riskApprovalEnabled: 'Enable approval threshold', riskApprovalKrw: 'Approval Threshold (KRW)', riskApprovalUsd: 'Approval Threshold (USD)',
    riskSaveBtn: 'Save',
    settingsSafetyTitle: 'Safety Controls',
    settingsSafetyHint: 'The kill switch is an emergency stop that immediately blocks all buys and sells. The daily-loss circuit breaker blocks new buys only when the loss against today\'s starting equity exceeds the limit (selling always stays available).',
    safetyKillSwitchLabel: 'Kill Switch',
    safetyDailyPnlDomestic: 'Today\'s P&L (Domestic, since 00:00 KST)',
    safetyDailyPnlOverseas: 'Today\'s P&L (Overseas, since 00:00 ET)',
    runtimeChangeFailed: msg => `Change failed: ${msg}`,
    runtimeChangeSaved: 'Changed · applied immediately.',
    runtimeChangeRequestFailed: 'Change request failed — check your connection.',
    riskCapZero: (label, cap) => `The ${label} order cap is ${cap} — a cap of 0 blocks every order. Enter an amount greater than 0.`,
    riskThresholdTooHigh: (label, thr, cap) => `The ${label} approval threshold (${thr}) must be lower than the order cap (${cap}). Orders above the cap are blocked regardless of approval, so a threshold at or above the cap has no effect.`,
    riskSaveFailed: msg => `Save failed: ${msg}`,
    riskSaved: 'Saved · applied to the order gate.',
    riskSaveRequestFailed: 'Save request failed — check your connection.',
    dailyLossUnavailable: 'No data',
    dailyLossBadge: (pnl, thr) => `${pnl}% (loss limit ${thr}%)`,
    dailyLossBreachedSuffix: ' — new buys blocked',
    safetyLoadFailed: 'Failed to load',
    killSwitchEngagedBadge: reason => `Stopped (${reason})`,
    killSwitchNoReason: 'no reason given',
    killSwitchOkBadge: 'Normal (trading enabled)',
    killSwitchDisengageBtn: 'Release Stop',
    killSwitchActionFailed: msg => `Failed: ${msg}`,
    killSwitchEngagedResult: 'Emergency stop engaged.',
    killSwitchDisengagedResult: 'Emergency stop released.',
    killSwitchRequestFailed: 'Request failed — check your connection.',
  },
};

export function t(key, ...args) {
  const v = I18N[LANG][key];
  return typeof v === 'function' ? v(...args) : v;
}

// Locale for human-readable dates/times (not numbers — share counts and
// prices stay 'en-US' digit grouping regardless of language, matching this
// dashboard's existing convention). Fixes a real inconsistency found in
// review: some date displays were hardcoded 'en-US' and others 'ko-KR'
// independent of each other, even within the Korean-only version.
export function dateLocale() { return LANG === 'ko' ? 'ko-KR' : 'en-US'; }

export function applyStaticI18n() {
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  document.querySelectorAll('[data-i18n-aria-label]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.i18nAriaLabel)); });

  const switchEl = document.getElementById('lang-switch');
  switchEl.innerHTML = `
    <a href="${langUrl('ko')}" class="lang-btn ${LANG === 'ko' ? 'active' : ''}">한</a>
    <a href="${langUrl('en')}" class="lang-btn ${LANG === 'en' ? 'active' : ''}">EN</a>
  `;
}
applyStaticI18n();

// Theme toggle: reads the effective theme (explicit data-theme attribute,
// falling back to the OS preference) rather than tracking its own state,
// so it stays correct even if the OS preference changes underneath it.
export function currentTheme() {
  const explicit = document.documentElement.getAttribute('data-theme');
  return explicit || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}
export function applyThemeButton() {
  const th = currentTheme();
  document.getElementById('theme-icon').textContent = th === 'dark' ? '☀️' : '🌙';
  document.getElementById('theme-toggle').setAttribute('aria-label', th === 'dark' ? t('themeLight') : t('themeDark'));
}
document.getElementById('theme-toggle').addEventListener('click', () => {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  applyThemeButton();
});
applyThemeButton();

// 국내/해외 scope — global (top header, matches the mockup), not tied to any
// one tab: 트레이딩's trade allowlist and AI 시그널's cards both filter against it.
export let _marketScope = localStorage.getItem('marketScope') === 'overseas' ? 'overseas' : 'domestic';
export function setMarketScope(scope) {
  _marketScope = scope;
  localStorage.setItem('marketScope', scope);
  document.querySelectorAll('#market-scope-toggle .seg-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.scope === scope);
  });
}
export function isDomesticSymbol(symbol) { return symbol.endsWith('.KS'); }

// 손절가 근접 기준 — 서버의 auto_trade_held_review_buffer_pct(장중 재검토를 강제하는
// 바로 그 값)를 /api/auto-trade/status에서 받아 보관. 프론트가 자체 기준을 만들면
// 하이라이트가 "시스템이 실제로 재검토하는 구간"과 어긋나므로 값을 지어내지 않고,
// 아직 못 받았으면 null로 두어 거리(%)만 보여주고 강조는 하지 않음. signals.js가
// 폴링할 때 채우고 trading.js가 읽으므로, 두 모듈이 서로 import하는 순환을 만들지
// 않도록 양쪽 다 의존하는 이 모듈에 둠.
export let _heldReviewBufferPct = null;
export function setHeldReviewBufferPct(pct) {
  _heldReviewBufferPct = typeof pct === 'number' && pct > 0 ? pct : null;
}

// 관심종목(trade allowlist) 서브탭(관심종목/실시간/최근 본) — 목업 매칭. "실시간"은
// 이미 있는 per-symbol market_open을 재사용하므로 별도 상태 불필요; "최근 본"만 아래
// 로컬 기록이 필요.
// localStorage key renamed 'watchlistSubTab' -> 'tradeAllowlistSubTab' (2026-08-05) —
// read the old key once as a fallback so an existing user's saved subtab choice isn't
// silently reset to the default the first time they load the renamed frontend.
export let _tradeAllowlistSubTab =
  localStorage.getItem('tradeAllowlistSubTab') || localStorage.getItem('watchlistSubTab') || 'favorites';
export function setTradeAllowlistSubTab(tab) {
  _tradeAllowlistSubTab = tab;
  localStorage.setItem('tradeAllowlistSubTab', tab);
  document.querySelectorAll('#trade-allowlist-subtabs .seg-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.subtab === tab);
  });
}

const RECENTLY_VIEWED_KEY = 'recentlyViewedSymbols';
const RECENTLY_VIEWED_MAX = 10;
export function getRecentlyViewed() {
  try { return JSON.parse(localStorage.getItem(RECENTLY_VIEWED_KEY)) || []; } catch { return []; }
}
export function pushRecentlyViewed(symbol) {
  const list = [symbol, ...getRecentlyViewed().filter(s => s !== symbol)].slice(0, RECENTLY_VIEWED_MAX);
  localStorage.setItem(RECENTLY_VIEWED_KEY, JSON.stringify(list));
}

export const state = {};   // symbol -> { name, previousClose, currency, history: [{price,volume,timestamp}], pct, canvas, tooltip }
export const MAX_HISTORY = 500;   // mirrors the server's own per-ticker tick buffer (app/live/store.py)

// Every list on the page shows at most LIST_CAP items until its 더보기
// button is pressed — the whole point of the redesign is glanceability,
// and an uncapped list quietly grows back into the old wall of data.
export const LIST_CAP = 5;
export const RANK_LIST_CAP = 10;  // rank screener gets its own, larger cap — user-requested
export const expanded = {};  // section key -> bool

export function moreButton(key, hiddenCount, rerender) {
  const btn = document.createElement('button');
  btn.className = 'more-btn';
  btn.textContent = expanded[key] ? t('collapse') : t('showMore', hiddenCount);
  btn.addEventListener('click', () => { expanded[key] = !expanded[key]; rerender(); });
  return btn;
}

export function fmtPrice(price, currency) {
  if (price == null) return '—';
  const decimals = currency === 'KRW' ? 0 : 2;
  // Sign goes before the currency symbol (-$14.87), not after ($-14.87) —
  // price.toLocaleString() puts its own minus sign in front of the digits,
  // so a negative price must be formatted from its absolute value with the
  // sign pulled out separately, same convention as fmtKRW/fmtUSD below.
  const sign = price < 0 ? '-' : '';
  const num = Math.abs(price).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  // Prefix the currency symbol so standalone amounts (호가·평단·손익·OHLC)
  // aren't ambiguous. ₩ for KRW, $ for USD; unknown currency stays bare.
  const symbol = currency === 'KRW' ? '₩' : currency === 'USD' ? '$' : '';
  return sign + symbol + num;
}

// `.KS` is a yfinance/KIS implementation detail (the KRX-ticker suffix this
// whole codebase uses internally), not something a user should ever have to
// read — strip it for DISPLAY ONLY. Never apply this to a value used as an
// identifier (data-symbol, cssSafe() ids, <option value>, fetch URLs, state[]
// lookups, ===-comparisons) — those all need the real ticker. The rank table
// used to do this inline (`r.ticker.replace('.KS','')`); centralized here so
// every display site shares one implementation instead of drifting apart.
export function displayTicker(ticker) {
  // Guard: called from inside table/card render loops, so one row with a
  // missing ticker must degrade to that one bad cell, not throw and blank
  // the whole table.
  if (typeof ticker !== 'string') return ticker;
  return ticker.replace(/\.KS$/i, '');
}

export function fmtDelta(pct) {
  const sign = pct > 0 ? '+' : '';
  return `${sign}${(pct * 100).toFixed(2)}%`;
}

// Tweens a price display from its previous value up/down to the new one
// over a short window instead of just snapping — the "scrolling numbers"
// effect from stock-ticker sites. Pure visual polish, no data implication:
// the intermediate values shown mid-animation are never real prices, just
// interpolated frames, same as any odometer-style counter. Steps on a plain
// setInterval rather than requestAnimationFrame deliberately — this is a
// background dashboard as often as a focused tab, and rAF callbacks don't
// fire at all in a backgrounded/inactive tab (verified live), which would
// just make prices silently stop animating instead of degrading gracefully.
export const _priceAnimations = new WeakMap();
export const _ANIMATION_STEP_MS = 16;

export function animateNumber(el, from, to, currency, durationMs = 500) {
  const existing = _priceAnimations.get(el);
  if (existing) clearInterval(existing);

  if (from == null || to == null || from === to) {
    el.textContent = fmtPrice(to, currency);
    return;
  }

  const flashClass = to > from ? 'flash-up' : 'flash-down';
  el.classList.remove('flash-up', 'flash-down');
  el.classList.add(flashClass);
  const start = Date.now();

  function step() {
    const t = Math.min((Date.now() - start) / durationMs, 1);
    const eased = 1 - Math.pow(1 - t, 3);  // ease-out cubic — fast start, gentle landing
    el.textContent = fmtPrice(from + (to - from) * eased, currency);
    if (t >= 1) {
      clearInterval(intervalId);
      _priceAnimations.delete(el);
      setTimeout(() => el.classList.remove(flashClass), 400);
    }
  }
  const intervalId = setInterval(step, _ANIMATION_STEP_MS);
  _priceAnimations.set(el, intervalId);
  step();
}

export function getVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function cssSafe(symbol) { return symbol.replace(/[^a-zA-Z0-9]/g, '_'); }

export function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// encodeURI() only percent-encodes special characters — it does NOT check
// the scheme, so a news item's link/thumbnail_url (third-party, unverified
// source) could be a "javascript:"/"data:" URI that runs when clicked
// (href) or acts as a tracking/phishing vector (img src). Only http(s) is
// ever a legitimate news link, so anything else is rejected outright.
// encodeURI만으로는 스킴 검증이 안 되어 javascript: 등 위험한 URI가 그대로 통과할 수 있음 —
// http/https만 허용.
export function safeUrl(url) {
  if (!url) return null;
  try {
    const parsed = new URL(url, window.location.href);
    return (parsed.protocol === 'http:' || parsed.protocol === 'https:') ? parsed.href : null;
  } catch {
    return null;
  }
}

export function formatTimeAgo(iso) {
  if (!iso) return '';
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (mins < 60) return t('minutesAgo', mins);
  const hours = Math.round(mins / 60);
  if (hours < 24) return t('hoursAgo', hours);
  return t('daysAgo', Math.round(hours / 24));
}

export function fmtCompact(value) {
  if (value == null) return '—';
  const abs = Math.abs(value);
  if (abs >= 1e9) return (value / 1e9).toFixed(2) + 'B';
  if (abs >= 1e6) return (value / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (value / 1e3).toFixed(1) + 'K';
  return value.toFixed(2);
}

export function fmtKRW(v, { signed = false } = {}) {
  if (v == null) return '—';
  const sign = v < 0 ? '−' : (signed && v > 0 ? '+' : '');
  return `${sign}₩${Math.abs(Math.round(v)).toLocaleString('en-US')}`;
}

export function fmtUSD(v, { signed = false } = {}) {
  if (v == null) return '—';
  const sign = v < 0 ? '−' : (signed && v > 0 ? '+' : '');
  return `${sign}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

// A bare HTTP status code ("정정 실패: 500") tells a user nothing actionable —
// only the codes this app's own endpoints actually return are mapped; a
// generic (non-numeric) fallback covers anything else so an unmapped code
// still shows *something* rather than printing the number bare. Callers
// should still prefer a server-supplied `detail` string when one exists —
// this is only the fallback for when there isn't one.
const _HTTP_ERROR_TEXT = {
  ko: {
    400: '요청이 올바르지 않습니다.',
    401: '인증이 필요합니다.',
    403: '권한이 없습니다.',
    404: '대상을 찾을 수 없습니다 (이미 처리됐을 수 있습니다).',
    409: '다른 처리와 충돌했습니다. 잠시 후 다시 시도하세요.',
    412: '주문 상태가 바뀌어 처리할 수 없습니다. 새로고침 후 다시 시도하세요.',
    422: '입력값을 처리할 수 없습니다.',
    429: '요청이 너무 많습니다. 잠시 후 다시 시도하세요.',
    500: '서버 오류가 발생했습니다.',
    502: '서버와 통신할 수 없습니다.',
    503: '서비스를 일시적으로 사용할 수 없습니다.',
    _default: '요청을 처리하지 못했습니다.',
  },
  en: {
    400: 'The request was invalid.',
    401: 'Authentication is required.',
    403: "You don't have permission for this.",
    404: 'Not found (it may already have been processed).',
    409: 'This conflicted with another update. Try again shortly.',
    412: 'The order state changed, so this could not be applied. Refresh and try again.',
    422: 'The submitted values could not be processed.',
    429: 'Too many requests. Try again shortly.',
    500: 'A server error occurred.',
    502: 'Could not reach the server.',
    503: 'The service is temporarily unavailable.',
    _default: 'The request could not be processed.',
  },
};
export function httpErrorText(status) {
  const table = _HTTP_ERROR_TEXT[LANG];
  return table[status] || table._default;
}

