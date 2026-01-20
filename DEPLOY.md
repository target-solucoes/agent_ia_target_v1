# Deploy Guide - Analytics Chatbot

## Pre-requisitos

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **API Key:**
  - Google Gemini API Key (obrigatoria)
  - Supabase (opcional, para logging)

## Arquitetura

A imagem Docker e **autossuficiente**:
- Dataset embutido durante o build
- Sem necessidade de upload manual de arquivos
- Configuracao apenas via variavel de ambiente `GEMINI_API_KEY`

## Deploy Local (Desenvolvimento)

### 1. Clonar e Configurar

```bash
# Clonar repositorio
git clone https://github.com/target-solucoes/analytics-chatbot.git
cd analytics-chatbot

# Criar arquivo de ambiente
cp .env.example .env
```

### 2. Configurar Variavel de Ambiente

Edite o arquivo `.env` com sua chave de API:

```bash
GEMINI_API_KEY=your-gemini-api-key-here
```

### 3. Build e Execucao

```bash
# Build e iniciar container
docker-compose up --build

# Ou em modo detached (background)
docker-compose up --build -d
```

### 4. Acessar Aplicacao

Abra o navegador em: http://localhost:8501

## Deploy em Producao (GHCR)

### 1. Autenticacao no Registry

```bash
# Criar token em: GitHub Settings > Developer settings > Personal access tokens
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

### 2. Pull da Imagem

```bash
docker pull ghcr.io/target-solucoes/analytics-chatbot:latest
```

### 3. Executar Container

```bash
docker run -d \
  --name analytics-chatbot \
  -p 8501:8501 \
  -e GEMINI_API_KEY=your-gemini-api-key \
  -v ./logs:/app/logs \
  -v ./data/output:/app/data/output \
  --restart unless-stopped \
  ghcr.io/target-solucoes/analytics-chatbot:latest
```

### 4. Ou usar Docker Compose

Crie um `docker-compose.prod.yml`:

```yaml
services:
  analytics-chatbot:
    image: ghcr.io/target-solucoes/analytics-chatbot:latest
    container_name: analytics-chatbot
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./data/output:/app/data/output
    restart: unless-stopped
```

Execute com:

```bash
GEMINI_API_KEY=your-key docker-compose -f docker-compose.prod.yml up -d
```

## Comandos de Manutencao

### Verificar Status

```bash
# Status do container
docker-compose ps

# Health check
curl http://localhost:8501/_stcore/health
```

### Logs

```bash
# Ver logs em tempo real
docker-compose logs -f analytics-chatbot

# Ultimas 100 linhas
docker-compose logs --tail=100 analytics-chatbot
```

### Reiniciar

```bash
# Restart simples
docker-compose restart

# Restart com rebuild
docker-compose down && docker-compose up --build -d
```

### Atualizar para Nova Versao

```bash
# Pull nova imagem
docker pull ghcr.io/target-solucoes/analytics-chatbot:latest

# Recriar container
docker-compose down
docker-compose up -d
```

### Limpar Recursos

```bash
# Parar e remover container
docker-compose down

# Remover imagens nao utilizadas
docker image prune -f

# Remover tudo (cuidado!)
docker-compose down --rmi all --volumes
```

## Validacao do Deploy

### 1. Health Check

```bash
# Deve retornar "ok"
curl -f http://localhost:8501/_stcore/health
```

### 2. Verificar Logs

```bash
# Verificar se nao ha erros de startup
docker-compose logs analytics-chatbot | grep -i error
```

### 3. Teste Funcional

1. Acesse http://localhost:8501
2. Faca login (se autenticacao habilitada)
3. Envie uma query de teste: "top 5 clientes por vendas"
4. Verifique se o grafico e gerado corretamente

## Troubleshooting

### Container nao inicia

```bash
# Verificar logs de erro
docker-compose logs analytics-chatbot

# Verificar se portas estao em uso
netstat -tulpn | grep 8501
```

### Erro de API Key

```bash
# Verificar se variavel foi carregada
docker-compose exec analytics-chatbot env | grep GEMINI
```

### Health check falhando

```bash
# Testar conectividade interna
docker-compose exec analytics-chatbot curl -f http://localhost:8501/_stcore/health
```

### Memoria insuficiente

Adicione limites de recursos no docker-compose.yml:

```yaml
services:
  analytics-chatbot:
    # ... outras configs
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## Seguranca

- **Nunca** commite o arquivo `.env` no git
- Container roda como usuario nao-root
- Healthcheck habilitados para deteccao de falhas
- GitHub Secrets para CI/CD
- Attestation de proveniencia da imagem

## Estrutura de Volumes

| Volume Local | Container | Modo | Descricao |
|--------------|-----------|------|-----------|
| `./logs` | `/app/logs` | rw | Logs da aplicacao |
| `./data/output` | `/app/data/output` | rw | Graficos exportados |

## Variaveis de Ambiente

| Variavel | Obrigatorio | Default | Descricao |
|----------|-------------|---------|-----------|
| `GEMINI_API_KEY` | Sim | - | API Key do Google Gemini |
| `DATASET_PATH` | Nao | `data/datasets/...parquet` | Caminho do dataset (embutido) |
| `SUPABASE_URL` | Nao | - | URL do Supabase |
| `SUPABASE_API_KEY` | Nao | - | API Key do Supabase |

## CI/CD com GitHub Actions

O workflow `.github/workflows/docker-publish.yml` automatiza:

1. **Push para `main`**: Build e push da imagem com tag `main`
2. **Tags `v*`**: Build e push com versao semantica
3. **Pull Requests**: Build apenas (sem push)

### Configurar GitHub Secret

No repositorio, va em Settings > Secrets and variables > Actions e adicione:

- `GEMINI_API_KEY` - Chave da API Gemini (para testes no CI se necessario)

### Tags Geradas

- `main` - Branch principal
- `v1.0.0` - Versao semantica
- `1.0` - Major.Minor
- `sha-abc1234` - Commit SHA

### Usar Versao Especifica

```bash
# Versao especifica
docker pull ghcr.io/target-solucoes/analytics-chatbot:v1.0.0

# Commit especifico
docker pull ghcr.io/target-solucoes/analytics-chatbot:sha-abc1234
```
