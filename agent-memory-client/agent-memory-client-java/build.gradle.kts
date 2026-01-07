plugins {
    id("java-library")
    id("maven-publish")
}

group = "com.redis"
version = project.findProperty("version") as String? ?: "0.1.0"
description = "Java client for the Agent Memory Server REST API"

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
    withJavadocJar()
    withSourcesJar()
}

repositories {
    mavenCentral()
}

dependencies {
    // HTTP Client
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // JSON Processing
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")

    // ULID generation
    implementation("com.github.f4b6a3:ulid-creator:5.2.3")

    // Annotations
    compileOnly("org.jetbrains:annotations:24.1.0")

    // Testing
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("org.mockito:mockito-core:5.8.0")
    testImplementation("org.mockito:mockito-junit-jupiter:5.8.0")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")

    // Testcontainers for integration tests
    testImplementation(platform("org.testcontainers:testcontainers-bom:1.19.3"))
    testImplementation("org.testcontainers:testcontainers")
    testImplementation("org.testcontainers:junit-jupiter")
}

tasks.test {
    useJUnitPlatform {
        excludeTags("integration")
    }
}

// Create a separate task for integration tests
tasks.register<Test>("integrationTest") {
    description = "Runs integration tests with Testcontainers"
    group = "verification"

    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath

    useJUnitPlatform {
        includeTags("integration")
    }

    shouldRunAfter(tasks.test)

    // Integration tests may take longer
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
}

tasks.javadoc {
    options {
        (this as StandardJavadocDocletOptions).apply {
            addStringOption("Xdoclint:none", "-quiet")
        }
    }
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])

            groupId = project.group.toString()
            artifactId = project.name
            version = project.version.toString()

            pom {
                name.set("Agent Memory Client Java")
                description.set(project.description)
                url.set("https://github.com/redis-developer/agent-memory-server")
                inceptionYear.set("2024")

                licenses {
                    license {
                        name.set("Apache License 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0")
                    }
                }

                developers {
                    developer {
                        id.set("redis")
                        name.set("Brian Sam-Bodden.")
                        email.set("bsbodden@redis.com")
                        organization.set("Redis Inc.")
                        organizationUrl.set("https://redis.io")
                    }
                }

                scm {
                    connection.set("scm:git:git://github.com/redis-developer/agent-memory-server.git")
                    developerConnection.set("scm:git:ssh://github.com:redis-developer/agent-memory-server.git")
                    url.set("https://github.com/redis-developer/agent-memory-server")
                }
            }
        }
    }

    repositories {
        maven {
            name = "staging"
            url = uri(layout.buildDirectory.dir("staging-deploy"))
        }
    }
}
