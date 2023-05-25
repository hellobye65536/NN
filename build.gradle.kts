plugins {
    id("java")
}

group = "hb"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("me.tongfei:progressbar:0.9.5")
}

task("runTraining", JavaExec::class) {
    mainClass.set("hb.app.Train")
    classpath = sourceSets["main"].runtimeClasspath
}
