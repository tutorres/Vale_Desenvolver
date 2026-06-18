# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


def set_text(paragraph, text):
    # Clear every w:t, including ones nested inside w:hyperlink (email fields
    # are hyperlink fields in this template — paragraph.runs only sees direct
    # w:r children, so it misses hyperlink-wrapped runs and would otherwise
    # append instead of replace).
    for t in paragraph._p.findall(".//" + qn("w:t")):
        t.text = ""
    for hyperlink in paragraph._p.findall(qn("w:hyperlink")):
        paragraph._p.remove(hyperlink)
    if paragraph.runs:
        paragraph.runs[0].text = text
    else:
        paragraph.add_run(text)


def insert_paragraph_after(paragraph, text="", style=None):
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style is not None:
        new_para.style = style
    if text:
        new_para.add_run(text)
    return new_para


doc = Document("data/Desenvolver_Template.docx")
p = doc.paragraphs

# --- Participantes ---
set_text(p[2], "")  # Nome do Grupo — trabalho individual
set_text(p[3], "Arthur Torres")
set_text(p[4], "CEFET-MG — Engenharia de Computação")
set_text(p[5], "arthuroliveiratorres@gmail.com")
set_text(p[7], "")
set_text(p[8], "")
set_text(p[9], "")
set_text(p[11], "")
set_text(p[12], "")
set_text(p[13], "")

# --- Resumo ---
resumo = (
    "Este trabalho desenvolve uma solução de análise preditiva para antecipar alertas "
    "críticos (Don't Go) em equipamentos de mineração — caminhões e escavadeiras — a "
    "partir de 37.164.054 registros de telemetria coletados entre janeiro e junho de "
    "2025. O desafio incluiu erros de qualidade de dados inseridos intencionalmente, "
    "dos quais 3 eram esperados (corrupção de encoding em Criticidade, string literal "
    "\"NULL\" em Classe, vírgula como separador decimal em Valor) e um quarto foi "
    "descoberto apenas ao processar o dataset completo: a mesma string \"NULL\" também "
    "aparecia na coluna Valor, afetando 237.443 registros.\n\n"
    "A solução foi estruturada como um pipeline ETL reprodutível em 5 notebooks "
    "numerados (limpeza, EDA/validação de regras de negócio, feature engineering, "
    "treinamento de modelo e relatório final), apoiados por 7 módulos Python testados "
    "(64 testes unitários, desenvolvidos com TDD). As features incluem contagens de "
    "alarmes em 5 janelas deslizantes (15min a 4h), tempo desde o último alarme "
    "crítico, tipo de equipamento e estado operacional — todas calculadas sem "
    "vazamento temporal. O modelo é um XGBoost com scale_pos_weight para o "
    "desbalanceamento extremo das classes (1,4% a 2,6% de positivos, conforme o "
    "horizonte), treinado e avaliado com split temporal (treino: janeiro-abril; "
    "teste: maio-junho) para 3 horizontes de previsão (1h, 2h, 4h).\n\n"
    "O horizonte de 4h apresentou o melhor F1-score (0,186 no threshold padrão; 0,207 "
    "no threshold 0,7), com recall de 57,7% e AUC de 0,767. A análise SHAP revelou que "
    "o tipo de equipamento é, isoladamente, a feature mais influente — mais que o "
    "dobro do impacto da segunda colocada (tempo desde o último alarme crítico) — "
    "sugerindo que o modelo captura majoritariamente um risco-base por tipo de "
    "equipamento, com o sinal dinâmico de alarmes contribuindo de forma secundária.\n\n"
    "Rodar o pipeline na escala real (37M linhas, 16GB de RAM disponíveis) expôs "
    "limitações de implementação que os testes unitários com dados sintéticos não "
    "capturavam: colunas de string carregadas como object excediam a memória "
    "disponível, e a mesma coluna podia reverter de category para object durante a "
    "concatenação de meses com vocabulários diferentes. Ambos os problemas foram "
    "corrigidos via TDD antes de prosseguir, junto com uma incompatibilidade conhecida "
    "entre as versões pinadas de shap e xgboost na etapa de explicabilidade. O "
    "resultado é um pipeline auditável, com cada decisão técnica documentada em "
    "spec/DECISIONS.md, e métricas reportadas com transparência — incluindo suas "
    "limitações de precisão — em vez de apenas o resultado mais favorável."
)
set_text(p[17], resumo)

set_text(p[18], "Palavras-Chave: Telemetria de Mineração; Manutenção Preditiva; XGBoost; Qualidade de Dados; SHAP")

# --- Introdução ---
set_text(p[20], (
    "Equipamentos de mineração — caminhões e escavadeiras — geram constantemente "
    "alarmes de telemetria que registram o estado operacional e eventos de risco. "
    "Entre eles estão os alertas \"Don't Go\": situações em que o equipamento "
    "acumula um número de alarmes de determinado tipo dentro de uma janela de tempo, "
    "indicando risco operacional que exige parada ou intervenção. Antecipar esses "
    "eventos, em vez de apenas reagir a eles, permite à operação planejar manutenção, "
    "reduzir paradas não programadas e mitigar riscos de segurança — exemplificando "
    "como dados de telemetria, normalmente usados para diagnóstico retroativo, podem "
    "sustentar decisões operacionais proativas."
))
set_text(p[22], (
    "O objetivo central deste trabalho é construir um pipeline de ETL robusto e "
    "reprodutível que limpe, valide e transforme os dados brutos de telemetria em um "
    "conjunto de features preditivas, e treinar um modelo de classificação capaz de "
    "prever a ocorrência de um Don't Go nas próximas 1h, 2h ou 4h, com avaliação "
    "honesta de desempenho (F1, precision, recall, AUC) e explicabilidade via SHAP."
))

# --- Entendimento do Negócio ---
set_text(p[24], (
    "Os dados de telemetria (37.164.054 registros, jan-jun/2025) registram cada "
    "alarme emitido por 35 equipamentos (TAGs), classificados por criticidade "
    "(crítico, não crítico, informacional) e tipo (caminhão ou escavadeira). "
    "Paralelamente, os apontamentos (377.907 registros) descrevem o estado "
    "operacional de cada equipamento ao longo do tempo (Operando, Parado, "
    "Hibernando, Manutenção). O sinal Is_Dont_Go, presente na telemetria, indica "
    "quando um equipamento atingiu o critério de parada definido pelas regras de "
    "negócio (136 regras, uma por tipo de alarme, definindo quantidade de "
    "ocorrências e janela de tempo). Esses dados existem porque a operação precisa "
    "rastrear, auditar e justificar paradas de equipamento — tanto por segurança "
    "quanto por eficiência de frota. Antecipar esse sinal, em vez de apenas "
    "registrá-lo, é o que transforma esses dados de um registro histórico em uma "
    "ferramenta de decisão: o time de operação pode agir antes da parada ocorrer, e "
    "não apenas explicá-la depois."
))

# --- Metodologia ---
set_text(p[26], (
    "O pipeline é dividido em 5 notebooks numerados e 7 módulos Python testados "
    "(src/loader.py, cleaner.py, validator.py, business_rules.py, features.py, "
    "model.py, utils.py — 64 testes unitários, desenvolvidos com TDD). A arquitetura "
    "segue: (1) carregamento e limpeza dos 6 arquivos mensais de telemetria e dos "
    "apontamentos; (2) EDA e validação das regras de negócio, reproduzindo o sinal "
    "Is_Dont_Go via janela deslizante e comparando com o sinal registrado; (3) "
    "feature engineering — contagens de alarmes em 5 janelas (15min, 30min, 1h, 2h, "
    "4h), tempo desde o último alarme crítico, tipo de equipamento e estado "
    "operacional, calculadas com pandas .rolling() e np.searchsorted (O(n log n) por "
    "equipamento, vetorizado — essencial para os 37M registros); (4) treinamento de "
    "um XGBoost com scale_pos_weight para o desbalanceamento de classes, split "
    "temporal (treino jan-abr, teste mai-jun, nunca aleatório, para não vazar "
    "informação do futuro) e avaliação em 3 horizontes; (5) explicabilidade via SHAP "
    "e consolidação do relatório final."
))
set_text(p[27], (
    "A principal dificuldade técnica foi escalar o pipeline das fixtures sintéticas "
    "usadas nos testes (5 a 15 linhas) para os 37M registros reais, em uma máquina "
    "com 16GB de RAM. Colunas de string carregadas como object (em vez de category) "
    "projetavam ~30GB de uso de memória; corrigido convertendo para category no "
    "carregamento. Um problema mais sutil surgiu na concatenação dos 6 meses: o "
    "pandas reverte category para object quando o conjunto de categorias difere "
    "entre os meses (TAGs e operadores que só aparecem em parte do ano) — corrigido "
    "unificando o CategoricalDtype antes do concat. O mesmo padrão de memória se "
    "repetiu na feature current_state (apenas 5 valores possíveis, mas calculada "
    "como object) e na cópia de colunas não utilizadas dentro de build_features. "
    "Cada um desses problemas foi reproduzido com um teste antes da correção (TDD), "
    "e a suíte completa (64 testes) permanece verde. Adicionalmente, o dataset real "
    "revelou um quarto erro de qualidade não documentado (string 'NULL' também em "
    "Valor, 237.443 registros) e expôs uma incompatibilidade de versão entre shap e "
    "xgboost na etapa de explicabilidade, documentados em spec/DECISIONS.md (D008)."
))

# --- Resultados e Discussões ---
set_text(p[29], (
    "O diagnóstico de qualidade identificou e corrigiu 4 problemas nos 37.164.054 "
    "registros de telemetria: 11 registros com corrupção de encoding em Criticidade, "
    "36.104.611 registros com string literal 'NULL' em Classe, 237.443 registros com "
    "a mesma string 'NULL' em Valor, e 821.849 registros com vírgula como separador "
    "decimal em Valor. A reprodução do sinal Is_Dont_Go via janela deslizante, "
    "validada em uma amostra de 50 mil registros contra as 136 regras de negócio, "
    "obteve precision de 0,12 e recall de 0,59 — indicando que as regras "
    "documentadas capturam a maioria dos casos reais, mas com taxa relevante de "
    "falsos positivos."
))
r29 = insert_paragraph_after(p[29], (
    "O modelo XGBoost foi treinado e avaliado para os 3 horizontes propostos, com "
    "split temporal (23.273.520 registros de treino, 13.890.534 de teste). O "
    "horizonte de 4 horas apresentou o melhor F1-score (0,186 no threshold 0,5; "
    "0,207 no threshold 0,7), superando 1h (F1=0,114) e 2h (F1=0,143) — mesmo com "
    "AUC mais baixo (0,767 vs. 0,883 em 1h), o que indica uma fronteira de decisão "
    "mais bem calibrada para a métrica de negócio (F1), ainda que a capacidade geral "
    "de ranqueamento (AUC) seja menor. Em todos os horizontes, o recall é "
    "consideravelmente maior que a precision (ex: recall=0,577 vs. precision=0,111 "
    "em 4h/threshold 0,5) — esperado dado o desbalanceamento extremo das classes e "
    "a escolha de scale_pos_weight, que privilegia não perder eventos reais de "
    "Don't Go ao custo de mais falsos positivos. Aumentar o threshold para 0,7 "
    "melhora a precision (de 0,111 para 0,126) com perda mínima de recall (de 0,577 "
    "para 0,574), sugerindo que esse ajuste é preferível ao padrão de 0,5 em "
    "produção."
), style="Normal")
insert_paragraph_after(r29, (
    "A análise SHAP para o horizonte de 4h (shap_plots/shap_bar_label_4h.png) "
    "revelou que tipo_caminhao (caminhão vs. escavadeira) é, isoladamente, a feature "
    "mais influente — mais que o dobro do impacto de time_since_last_critical, a "
    "segunda colocada, e muito acima das contagens de alarme em qualquer janela. "
    "Esse resultado é uma limitação relevante a ser discutida com transparência: "
    "sugere que o modelo aprendeu majoritariamente uma diferença de risco-base entre "
    "os dois tipos de equipamento, em vez de um padrão dinâmico fino de comportamento "
    "de alarmes antes de um Don't Go. As features de contagem de alarme contribuem, "
    "mas com impacto bem menor — alarm_count_4h é a terceira mais relevante, seguida "
    "por uma queda abrupta nas janelas mais curtas (30m, 1h, 2h, 15m), coerente com "
    "o horizonte de 4h ser o melhor: alarmes acumulados em uma janela mais longa "
    "carregam mais sinal preditivo para esse horizonte do que rajadas de curto prazo."
), style="Normal")

# --- Conclusão e Trabalhos Futuros ---
set_text(p[31], (
    "O pipeline desenvolvido demonstra que é possível antecipar ocorrências de "
    "Don't Go com até 4 horas de antecedência, usando exclusivamente features "
    "derivadas do histórico de alarmes de telemetria, com um processo de ETL "
    "auditável e testado (64 testes, TDD) que escala dos dados sintéticos de teste "
    "até os 37 milhões de registros reais. As métricas obtidas — F1 entre 0,11 e "
    "0,21 — são modestas em termos absolutos, mas refletem honestamente um problema "
    "com desbalanceamento extremo (abaixo de 3% de positivos) e uma feature "
    "dominante (tipo de equipamento) que sugere espaço real para melhoria nas "
    "features dinâmicas antes de uma eventual adoção em produção."
))
set_text(p[32], (
    "Investigar por que tipo_caminhao domina o SHAP: treinar modelos separados por "
    "tipo de equipamento, ou adicionar interações explícitas, para isolar o sinal "
    "dinâmico de alarmes do risco-base por tipo."
))
b2 = insert_paragraph_after(p[32], (
    "Comparar XGBoost com LightGBM nos 3 horizontes, e testar SMOTE como ablação ao "
    "scale_pos_weight."
), style="List Bullet")
b3 = insert_paragraph_after(b2, (
    "Adicionar features de turno, frequência de manutenções recentes e localidade."
), style="List Bullet")
b4 = insert_paragraph_after(b3, (
    "Construir um dashboard (ex: Streamlit) para consulta de risco por equipamento "
    "em tempo real."
), style="List Bullet")
insert_paragraph_after(b4, (
    "Monitorar concept drift na distribuição das features entre os períodos de "
    "treino e teste."
), style="List Bullet")

doc.save("Relatorio_Final_Vale_Desenvolver2026.docx")
print("Saved Relatorio_Final_Vale_Desenvolver2026.docx")
