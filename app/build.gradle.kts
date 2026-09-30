plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "io.github.alexeyw.zimage"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "io.github.alexeyw.zimage"
        minSdk = 31
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        // The graphs need a 64-bit process (a 3.5 GB text encoder, ~1 GB DiT shards).
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Sample convenience: a release build installs without a keystore setup.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }

    testOptions { unitTests.isReturnDefaultValues = true }
}

tasks.withType<Test>().configureEach {
    // The parity tests read the upstream tables from models/host when present and skip otherwise.
    // Declared as inputs so the build cache cannot replay a "skipped" result once they exist.
    inputs.files(
        rootProject.fileTree("models/host") { include("vocab.json", "merges.txt", "t_emb_*.f32") },
    ).withPropertyName("hostAssets").withPathSensitivity(PathSensitivity.RELATIVE)
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.okhttp)
    implementation(libs.litert)
    debugImplementation(libs.androidx.compose.ui.tooling)

    testImplementation(libs.junit)
    // android.jar's org.json is a stub under JVM unit tests; the real one parses the fixtures.
    testImplementation(libs.json)
}
