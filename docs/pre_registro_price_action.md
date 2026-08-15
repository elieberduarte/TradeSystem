# Pré-registro: bateria de price action diário com magnitude

**Data do registro: 15/08/2026 — escrito ANTES de rodar qualquer teste.**

Regras do jogo, fixadas agora: escala diária (fricção ≈ 0,01 ATR);
tudo normalizado pelo ATR(14) do dia anterior; desfecho padrão =
retorno do dia seguinte (e acumulado D+1..D+3) em ATRs, P(sobe),
comparados à incondicional; 10 hipóteses × 2 símbolos (WIN$N, WDO$N)
= régua de Bonferroni em p < 0,005; célula sobrevivente vai ao teste
de replicação nos 28 instrumentos antes de qualquer conclusão.
Amostra mínima por célula: 30 (senão a linha reporta "sem amostra").

Notação: `r` = variação do fechamento em ATRs · `d` = deslocamento
abertura→fechamento em ATRs · `range` = máxima−mínima do dia em ATRs
· `pos` = posição do fechamento dentro do range do dia (0 = mínima,
1 = máxima).

---

## A. Anatomia do candle

**H1 — Pavio de rejeição com magnitude (o "martelo de verdade").**
Dia cuja mínima cai ≥ 1 ATR abaixo do fechamento de ontem MAS que
fecha com pos ≥ 0,67. Contraste: queda ≥ 1 ATR fechando com pos ≤ 0,33
(capitulação sem defesa). O dia seguinte difere entre os dois?
*Palpite: o martelo NÃO supera a capitulação — nossos estudos de
candle nunca acharam informação no desenho; a reversão média vem da
QUEDA, não do pavio.*

**H2 — Momentum de fechamento (anti-IBS).** Fechamento no decil
extremo do range (pos ≥ 0,9 / ≤ 0,1), qualquer direção do dia. O dia
seguinte continua o lado do fechamento?
*Palpite: leve continuação vendedora (pos ≤ 0,1 → amanhã cai), efeito
< 0,05 ATR; o espelho comprador mais fraco (deriva positiva de fundo).*

**H3 — Range gigante × range anão.** Dia com range ≥ 2 ATR vs dia com
range ≤ 0,5 ATR. Mede o dia seguinte em DIREÇÃO (continua o lado do
range gigante?) e em EXPANSÃO (range de amanhã em ATRs).
*Palpite: range anão → amanhã expande (compressão-expansão, tese do
squeeze validada no diário); range gigante → direção = moeda, range
de amanhã ainda elevado (clustering de vol).*

## B. Gaps com consequência

**H4 — Gap and go.** Abertura com gap ≥ 0,5 ATR que NÃO fecha o gap no
próprio dia (mínima do dia não alcança o fechamento de ontem, no gap
de alta). Acumulado D+1..D+3 continua na direção do gap?
*Palpite: leve continuação (~+0,05 ATR em 3 dias) — o gap não fechado
é seleção de força; mas metade disso deve sumir na correção de
Bonferroni.*

## C. Estruturas de 2-4 dias

**H5 — Bandeira de dois candles.** Deslocamento forte (|d| ≥ 1 ATR)
seguido de inside day. O rompimento sai do lado do impulso? Medir
direção de D+2 e do acumulado D+2..D+4.
*Palpite: sim, ~55% a favor do impulso — a única família (compressão
pós-impulso) que replicou em tudo que testamos.*

**H6 — Acumulação silenciosa.** Três fechamentos consecutivos com
pos ≥ 0,6 (compradores fechando forte três dias seguidos, sem exigir
alta grande). O 4º dia?
*Palpite: nada (moeda) — é o tipo de padrão bonito que a moeda produz
sozinha; ver micro-tendências.*

**H7 — O falso rompimento do Donchian (o anti-titular).** Máxima do
dia FURA a máxima dos 20 pregões anteriores mas o FECHAMENTO volta
para dentro do canal. O dia seguinte reverte (cai)?
*Palpite: reversão pequena (−0,03 a −0,06 ATR) — é o único candidato
de "pavio como informação" com mecanismo plausível (stops de
rompedores presos). Se der grande, é a descoberta da bateria; a
implicação operacional seria um FILTRO no titular, não um setup novo.*

## D. Localização no mapa maior

**H8 — A queda no fundo × a queda no meio.** Queda forte (r ≤ −1) com
fechamento no quartil INFERIOR do canal de 20 dias vs a mesma queda no
meio do canal (quartis 2-3). A reversão média do dia seguinte é maior
no fundo?
*Palpite: sim — reversão + localização devem interagir (a queda no
fundo encontra os compradores de valor); diferença ~0,05 ATR.*

## E. As extensões das perguntas do usuário

**H9 — O "V" de um dia.** Queda forte hoje, recuperação TOTAL amanhã
(fechamento acima do fechamento de anteontem — o raro 1-6%). O que o
3º dia faz?
*Palpite declarado com aviso: n ≈ 5-15, provavelmente "sem amostra";
se houver, continuação da recuperação (V é informação institucional).*

**H10 — Fundo duplo objetivo.** Mínima de hoje a ≤ 0,15 ATR da mínima
dos últimos 10 pregões, formada há 3-7 dias, SEM romper (fechamento
acima dela). O dia seguinte repica?
*Palpite: moeda (48-52%) — toda a família "defesa de nível" morreu no
random walk; o diário não deve salvá-la.*

---

## O que já está decidido sobre a leitura (antes dos números)

1. Célula com p < 0,005 E efeito ≥ 0,03 ATR → vai à replicação nos 28.
2. Célula significativa só em UM símbolo → suspeita de sorte de
   instrumento (band_fade nunca mais).
3. Nenhuma célula sobrevivente → a bateria vira documentação de que a
   anatomia diária, além da reversão-à-média já conhecida (família
   IBS), não carrega informação — e o capítulo fecha como o intraday.
4. Os palpites acima são falsificáveis de propósito. Placar do autor
   até aqui no projeto: 3 palpites certos, 3 falsificados.
