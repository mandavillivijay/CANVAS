{{- define "canvas-heal.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "canvas-heal.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{ include "canvas-heal.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "canvas-heal.selectorLabels" -}}
app.kubernetes.io/name: canvas-heal
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
