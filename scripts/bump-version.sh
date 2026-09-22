#!/usr/bin/env bash
set -euo pipefail

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Usage: $0 <major.minor.patch>"
  echo "  e.g. $0 1.2.0"
  exit 1
fi

# pyproject.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Helm Chart
sed -i "s/^version: .*/version: $VERSION/" deploy/k8s/helm/Chart.yaml
sed -i "s/^appVersion: .*/appVersion: \"$VERSION\"/" deploy/k8s/helm/Chart.yaml

# K8s manifests (if they have version)
# sed -i "s/image: .*:v.*/image: zdm-backend:$VERSION/" deploy/k8s/manifests/backend.yaml

echo "Bumped version to $VERSION"
echo "Next steps:"
echo "  1. git add -A && git commit -m \"chore: bump version to $VERSION\""
echo "  2. git tag v$VERSION"
echo "  3. git push origin v$VERSION"
echo "  This will trigger the release workflow."
