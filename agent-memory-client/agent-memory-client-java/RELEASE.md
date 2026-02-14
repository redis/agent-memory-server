# Release Process for Agent Memory Client Java

This document describes the release process for the Agent Memory Client Java library using JReleaser.

## Prerequisites

Before you can release, you need to set up the following GitHub secrets in your repository:

### Required Secrets

1. **GPG Signing Keys**
   - `GPG_PUBLIC_KEY` - Your GPG public key (armored format)
   - `GPG_SECRET_KEY` - Your GPG private key (armored format)
   - `GPG_PASSPHRASE` - The passphrase for your GPG key

2. **Maven Central Credentials**
   - `SONATYPE_USERNAME` - Your Sonatype username
   - `SONATYPE_PASSWORD` - Your Sonatype password (or token)

3. **GitHub Tokens**
   - `GITHUB_TOKEN` - Automatically provided by GitHub Actions (no setup needed)
   - `GIT_ACCESS_TOKEN` - Personal access token with write access to push tags (must be added as a repository secret)

### Setting Up GPG Keys

If you don't have GPG keys yet:

```bash
# Generate a new GPG key
gpg --full-generate-key

# Export your public key (armored format)
gpg --armor --export YOUR_EMAIL > public.key

# Export your private key (armored format)
gpg --armor --export-secret-keys YOUR_EMAIL > private.key
```

Then add the contents of these files to your GitHub secrets.

### Setting Up Maven Central

1. Create an account at https://central.sonatype.com/
2. Generate a user token from your account settings
3. Add the username and password to GitHub secrets

## Release Process

### 1. Prepare for Release

Ensure all changes are committed and pushed to the main branch:

```bash
git checkout main
git pull origin main
```

### 2. Trigger the Release

The release is triggered via GitHub Actions workflow dispatch:

1. Go to your repository on GitHub
2. Navigate to **Actions** → **Release Java Client**
3. Click **Run workflow**
4. Enter the version number (e.g., `0.2.0`, `1.0.0-RC.1`)
5. Click **Run workflow**

### 3. What Happens During Release

The workflow will:

1. **Create Version Tag**: Use the Axion Release Plugin to create a git tag (prefixed with `java-client-v`) for the specified version
2. **Build**: Compile and test the project
3. **Publish**: Publish artifacts to the staging repository
4. **Sign**: Sign all artifacts with GPG
5. **Assemble**: Prepare release artifacts with JReleaser
6. **Release**: 
   - Create a GitHub release with changelog
   - Deploy to Maven Central
   - Tag the repository

### 4. Verify the Release

After the workflow completes:

1. **Check GitHub Release**: Verify the release appears on GitHub with proper changelog
2. **Check Maven Central**: Wait ~10-30 minutes for artifacts to appear on Maven Central
   - Search at: https://central.sonatype.com/artifact/com.redis/agent-memory-client-java
3. **Test the Release**: Try using the new version in a test project

## Version Numbering

Follow semantic versioning (SemVer):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.2.0): New features, backward compatible
- **Patch** (0.1.1): Bug fixes, backward compatible

Pre-release versions:
- **RC** (1.0.0-RC.1): Release candidate
- **SNAPSHOT** (1.0.0-SNAPSHOT): Development snapshot
- **alpha/beta** (1.0.0-alpha.1): Early testing versions

## Troubleshooting

### Build Fails

Check the test reports artifact uploaded by the workflow.

### Signing Fails

Verify your GPG secrets are correctly set:
- Keys should be in armored format (begin with `-----BEGIN PGP`)
- Passphrase should match your GPG key

### Maven Central Deployment Fails

- Verify your Sonatype credentials
- Ensure your account has access to the `com.redis` groupId
- Check that all required POM metadata is present

### JReleaser Fails

Check the JReleaser output artifact for detailed logs:
- `out/jreleaser/trace.log`
- `out/jreleaser/output.properties`

## Manual Release (Local)

If you need to release manually from your local machine:

```bash
# Set environment variables
export JRELEASER_PROJECT_VERSION=0.2.0
export JRELEASER_GITHUB_TOKEN=your_token
export JRELEASER_GPG_PASSPHRASE=your_passphrase
export JRELEASER_GPG_PUBLIC_KEY="$(cat public.key)"
export JRELEASER_GPG_SECRET_KEY="$(cat private.key)"
export JRELEASER_MAVENCENTRAL_USERNAME=your_username
export JRELEASER_MAVENCENTRAL_PASSWORD=your_password

# Build and publish
./gradlew build test publish

# Run JReleaser
./gradlew jreleaserFullRelease
```

## Configuration Files

- **jreleaser.yml**: JReleaser configuration
- **build.gradle.kts**: Gradle build configuration with signing and Axion release plugin
- **.github/workflows/release-java-client.yml**: GitHub Actions workflow

## References

- [JReleaser Documentation](https://jreleaser.org/)
- [Maven Central Publishing Guide](https://central.sonatype.org/publish/)
- [GPG Signing Guide](https://central.sonatype.org/publish/requirements/gpg/)

