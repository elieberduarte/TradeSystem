//+------------------------------------------------------------------+
//| DonchianChannel.mq5 — o canal do TradeSystem, fiel ao bot         |
//|                                                                   |
//| Richard Donchian (anos 1950): máxima e mínima dos últimos N       |
//| períodos. É o gatilho da estratégia titular da carteira: compra   |
//| quando o FECHAMENTO supera a máxima dos 20 pregões ANTERIORES.    |
//|                                                                   |
//| InpShift=true (padrão) desenha o canal das N barras anteriores,   |
//| excluindo a atual — exatamente o que o bot compara com o          |
//| fechamento. Com false, inclui a barra atual (o desenho clássico   |
//| dos livros, que nunca pode ser "rompido" pela própria barra).     |
//+------------------------------------------------------------------+
#property copyright "TradeSystem"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots   3

#property indicator_label1  "Donchian superior"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrGoldenrod
#property indicator_style1  STYLE_DASH
#property indicator_width1  1

#property indicator_label2  "Donchian inferior"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrGoldenrod
#property indicator_style2  STYLE_DASH
#property indicator_width2  1

#property indicator_label3  "Meio do canal (Kijun do Ichimoku, se N=26)"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrGray
#property indicator_style3  STYLE_DOT
#property indicator_width3  1

input int  InpChannel = 20;    // Períodos do canal (o bot usa 20 no diário)
input bool InpShift   = true;  // Canal das N barras ANTERIORES (como o bot vê)

double BufferUpper[];
double BufferLower[];
double BufferMiddle[];

int OnInit()
  {
   SetIndexBuffer(0, BufferUpper,  INDICATOR_DATA);
   SetIndexBuffer(1, BufferLower,  INDICATOR_DATA);
   SetIndexBuffer(2, BufferMiddle, INDICATOR_DATA);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("Donchian(%d)%s", InpChannel,
                                   InpShift ? " [como o bot]" : ""));
   return(INIT_SUCCEEDED);
  }

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   int start = (prev_calculated > 0) ? prev_calculated - 1 : 0;

   for(int i = start; i < rates_total; i++)
     {
      int last  = InpShift ? i - 1 : i;          // última barra do canal
      int first = last - InpChannel + 1;
      if(first < 0)
        {
         BufferUpper[i]  = EMPTY_VALUE;
         BufferLower[i]  = EMPTY_VALUE;
         BufferMiddle[i] = EMPTY_VALUE;
         continue;
        }
      double hi = high[first];
      double lo = low[first];
      for(int j = first + 1; j <= last; j++)
        {
         if(high[j] > hi) hi = high[j];
         if(low[j]  < lo) lo = low[j];
        }
      BufferUpper[i]  = hi;
      BufferLower[i]  = lo;
      BufferMiddle[i] = (hi + lo) / 2.0;
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+
