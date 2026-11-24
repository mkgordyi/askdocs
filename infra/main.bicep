// Everything the service needs in Azure, described as code.
// Applied by the pipeline on every deploy, so the cloud always matches this file.

@description('Container image to run, for example ghcr.io/owner/askdocs:commit')
param image string

param location string = resourceGroup().location
param appName string = 'askdocs-api'

@description('0 lets the app sleep when nobody uses it. First request then takes a few seconds.')
@minValue(0)
@maxValue(1)
param minReplicas int = 0

@allowed(['none', 'openai', 'azure_openai'])
param llmProvider string = 'none'
param llmModel string = ''
param azureOpenAiEndpoint string = ''

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  // The app gets its own identity. With it the app can call Azure OpenAI without any key.
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'ASKDOCS_LLM_PROVIDER', value: llmProvider }
            { name: 'ASKDOCS_LLM_MODEL', value: llmModel }
            { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAiEndpoint }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: { path: '/health', port: 8080 }
              periodSeconds: 5
              failureThreshold: 24
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8080 }
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: 1
      }
    }
  }
}

output url string = 'https://${app.properties.configuration.ingress.fqdn}'
output principalId string = app.identity.principalId
