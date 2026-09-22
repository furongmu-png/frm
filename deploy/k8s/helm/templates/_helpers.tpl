{{/*
Expand the name of the chart.
*/}}
{{- define "zdm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name. Truncated to 63 chars to satisfy DNS-1123.
*/}}
{{- define "zdm.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Chart name and version label, used by commonLabels.
*/}}
{{- define "zdm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels applied to every resource.
*/}}
{{- define "zdm.labels" -}}
helm.sh/chart: {{ include "zdm.chart" . }}
{{ include "zdm.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels — must be unique and stable across upgrades.
*/}}
{{- define "zdm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "zdm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Backend-specific selector labels.
*/}}
{{- define "zdm.backendSelectorLabels" -}}
app.kubernetes.io/name: {{ include "zdm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: backend
{{- end -}}

{{/*
Frontend-specific selector labels.
*/}}
{{- define "zdm.frontendSelectorLabels" -}}
app.kubernetes.io/name: {{ include "zdm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: frontend
{{- end -}}

{{/*
The namespace to deploy into. Defaults to the release namespace unless
namespaceOverride is set.
*/}}
{{- define "zdm.namespace" -}}
{{- if .Values.namespaceOverride -}}
{{- .Values.namespaceOverride -}}
{{- else -}}
{{- .Release.Namespace -}}
{{- end -}}
{{- end -}}

{{/*
Service account name — uses the override or generates one from the release.
*/}}
{{- define "zdm.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "zdm.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Backend image reference (repository:tag).
*/}}
{{- define "zdm.backendImage" -}}
{{- printf "%s:%s" .Values.backend.image.repository .Values.backend.image.tag -}}
{{- end -}}

{{/*
Frontend image reference (repository:tag).
*/}}
{{- define "zdm.frontendImage" -}}
{{- printf "%s:%s" .Values.frontend.image.repository .Values.frontend.image.tag -}}
{{- end -}}
