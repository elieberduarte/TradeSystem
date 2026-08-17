//+------------------------------------------------------------------+
//| CandleTimer.mq5 — tempo restante para o fechamento do candle      |
//|                                                                   |
//| Mostra, no canto do gráfico, quanto falta para a barra atual do   |
//| tempo gráfico ativo fechar (mm:ss). Atualiza a cada segundo via   |
//| OnTimer, então funciona mesmo quando o mercado está parado (sem   |
//| tick novo). Muda de cor nos últimos segundos, opcionalmente.       |
//+------------------------------------------------------------------+
#property copyright "TradeSystem"
#property version   "1.00"
#property indicator_chart_window
#property indicator_plots 0

input int    InpCorner      = CORNER_RIGHT_UPPER; // Canto (0=SE,1=SD,2=IE,3=ID)
input int    InpX           = 12;                 // Deslocamento X (px)
input int    InpY           = 24;                 // Deslocamento Y (px)
input int    InpFontSize    = 12;                 // Tamanho da fonte
input color  InpColor       = clrWhite;           // Cor normal
input color  InpWarnColor   = clrOrange;          // Cor nos últimos segundos
input int    InpWarnSeconds = 15;                 // A partir de quantos segundos avisar

string g_name = "TradeSystem_CandleTimer";

int OnInit()
  {
   ObjectCreate(0, g_name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, g_name, OBJPROP_CORNER, InpCorner);
   ObjectSetInteger(0, g_name, OBJPROP_XDISTANCE, InpX);
   ObjectSetInteger(0, g_name, OBJPROP_YDISTANCE, InpY);
   ObjectSetInteger(0, g_name, OBJPROP_FONTSIZE, InpFontSize);
   ObjectSetString (0, g_name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, g_name, OBJPROP_COLOR, InpColor);
   ObjectSetInteger(0, g_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, g_name, OBJPROP_HIDDEN, true);
   EventSetTimer(1);
   Refresh();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectDelete(0, g_name);
  }

void OnTimer() { Refresh(); }

int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
  {
   Refresh();
   return(rates_total);
  }

void Refresh()
  {
   datetime bar_open = iTime(_Symbol, _Period, 0);
   if(bar_open == 0) return;
   int period_sec = PeriodSeconds(_Period);
   // TimeCurrent() = hora do servidor (mesma base do carimbo dos candles)
   long remaining = (long)(bar_open + period_sec) - (long)TimeCurrent();
   if(remaining < 0) remaining = 0;

   int hh = (int)(remaining / 3600);
   int mm = (int)((remaining % 3600) / 60);
   int ss = (int)(remaining % 60);
   string text = (hh > 0)
      ? StringFormat("%s  %02d:%02d:%02d", TimeframeLabel(), hh, mm, ss)
      : StringFormat("%s  %02d:%02d", TimeframeLabel(), mm, ss);

   ObjectSetString (0, g_name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, g_name, OBJPROP_COLOR,
                    (remaining <= InpWarnSeconds) ? InpWarnColor : InpColor);
   ChartRedraw();
  }

string TimeframeLabel()
  {
   string s = EnumToString(_Period);      // "PERIOD_M5"
   StringReplace(s, "PERIOD_", "");
   return s;
  }
//+------------------------------------------------------------------+
