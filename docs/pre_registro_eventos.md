# Pré-registro: estudo de eventos (PEAD) na B3

**Registrado em 17/08/2026, antes de rodar qualquer número.**

## Pergunta

Depois de um evento corporativo, o mercado reage no dia seguinte — e
**sub-reage**: a direção da reação inicial persiste nos dias 2 a 15?
É o *post-earnings announcement drift* (Ball & Brown 1968; Bernard &
Thomas 1989), a anomalia mais replicada das finanças, sem teste
público na B3 com régua honesta.

## Dados

- Eventos: IPE da CVM (fatos relevantes + "Dados Econômico-Financeiros"
  = comunicados de resultado), com **data de entrega** (o momento em que
  o mercado soube), para o maior conjunto de tickers com preço no
  acervo (13 originais + os líquidos do censo que tiverem CNPJ no FCA).
- Preços: diário do acervo (XP), 2021-08 → 2026-08.
- Point-in-time: evento entregue no dia D **após o fechamento** conta
  como reação em D+1; entregue **antes/durante o pregão** conta em D.
  A IPE não traz hora → **convenção conservadora**: reação = retorno
  de D+1 sobre D (assume que o mercado só precifica no pregão seguinte
  à entrega). Isso ENFRAQUECE o sinal se parte da reação ocorreu em D;
  é o erro do lado seguro.

## Definições

- `r_reacao` = retorno de fechamento D → D+1, em ATR(14) do papel.
- `r_deriva` = retorno acumulado D+1 → D+15 (14 pregões), em ATR.
- Reação "forte" = |r_reacao| ≥ 1 ATR (o evento moveu o papel).
- Deriva **a favor** = sign(r_deriva) == sign(r_reacao).

## Hipóteses (com palpite falsificável)

**H1 — Deriva geral.** Após reação forte, `r_deriva` na direção da
reação tem média > 0 e P(a favor) > 50%.
*Palpite: sim, fraco — P ≈ 53–56%, média +0,1 a +0,3 ATR. Se ≥ 58%
com n ≥ 200, é forte.*

**H2 — Resultados vs fatos relevantes.** A deriva é maior após
comunicados de RESULTADO (a categoria da literatura) que após fatos
relevantes genéricos.
*Palpite: sim — resultados são o evento "limpo"; fatos relevantes
misturam de tudo (recompra, aquisição, mudança de diretoria).*

**H3 — Assimetria.** Deriva após reação NEGATIVA é maior que após
positiva (má notícia é digerida devagar; instituições vendem em dias).
*Palpite: sim, moderado.*

**H4 — Placebo.** Datas aleatórias do mesmo papel (mesma quantidade,
mesmos anos), com o mesmo filtro |r| ≥ 1 ATR: a "deriva" após um dia
forte QUALQUER é a mesma? Se sim, o efeito é momentum de curto prazo,
não informação do evento.
*Palpite: o placebo mostra deriva PERTO DE ZERO (a bateria de price
action diário achou reversão fina, não continuação, após dias fortes
sem evento) — a diferença evento − placebo é o valor da notícia.*

## Régua

- Célula vale se p < 0,01 (poucas células: 4 hipóteses × 2 lados) E
  |efeito| ≥ 0,10 ATR (a fricção diária é ~0,01–0,03 ATR em ações
  líquidas; 0,10 dá folga para a construção do trade).
- Replicação: fração de papéis com deriva média > 0 no lado testado
  (≥ 60% com ≥ 15 papéis).
- Sobrevivente vira estratégia no motor (entrada no fechamento de
  D+1, saída em D+15 ou stop 2×ATR) e passa pela esteira normal.

## Se nada sobreviver

O buscador de notícias vira ferramenta INFORMATIVA (painel com o
feed CVM + RSS por ticker), sem promessa de edge — e o capítulo fica
documentado como os outros.
