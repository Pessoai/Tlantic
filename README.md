# Tlantic
DSBA final project

In this project, we will explore a dataset with the following columns:

LOJA_ID, LOJA, CIDADE, REGIAO, PRODUTIVIDADE_HORA, TOTAL_COLABORADORES, SKUS, CAIXAS_TRADICIONAIS, SELF_CHECKOUT, ABERTURA_LOJA, FECHO_LOJA, DATA_VENDA, ITEMS, VALOR_VENDA, SKUS_UP, SKUS_DOWN, AUMENTO_PRECO, DESCIDA_PRECO, DATA_FERIADO, TIPO_FERIADO, FERIADO_FIXO, ABERTURA_FERIADO, NOME_EVENTO, INICIO_EVENTO, FIM_EVENTO.

The final goal of this machine learning exercise is to create a model to predict the sales for a big supermarket company for the next 4 to 6 days. 

Some pending questions:
  - how to handle sales amounts on days when the store was closed;
      - should we sum the sales amount to the previous day?
      - after filling in missing values, should we delete  or set to 0 all lines for the 25/12 and 01/01?

