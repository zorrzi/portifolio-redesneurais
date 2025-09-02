## Setup

Para utilizar o código deste repositório, siga as instruções a seguir:

Crie um ambiente virtual do Python:

``` shell
python3 -m venv .venv
```

Ative o ambiente virtual (**você deve fazer isso sempre que for executar algum script deste repositório**):

``` shell
 .venv\Scripts\activate

```

Instale as dependências com:

``` shell
python3 -m pip install -r requirements.txt --upgrade
```

## Deployment

O material utiliza o [mkdocs](https://www.mkdocs.org/) para gerar a documentação. Para visualizar a documentação, execute o comando:

``` shell
mkdocs serve -o
```

Para subir ao GitHub Pages, execute o comando:

``` shell
mkdocs gh-deploy
```
