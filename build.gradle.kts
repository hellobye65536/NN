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
//    testImplementation("org.junit.jupiter:junit-jupiter:5.7.1")
//    implementation("org.nd4j:nd4j-native-platform:1.0.0-M1")
}

//tasks {
//    test {
//        useJUnitPlatform();
//    }
//}

task("runTraining", JavaExec::class) {
    mainClass.set("hb.app.Train")
    classpath = sourceSets["main"].runtimeClasspath
}
