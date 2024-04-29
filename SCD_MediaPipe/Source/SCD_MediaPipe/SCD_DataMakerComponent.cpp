// Fill out your copyright notice in the Description page of Project Settings.


#include "SCD_DataMakerComponent.h"

// Sets default values for this component's properties
USCD_DataMakerComponent::USCD_DataMakerComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void USCD_DataMakerComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}


// Called every frame
void USCD_DataMakerComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

void USCD_DataMakerComponent::SaveStringArrayToTxt(TArray<FString> Data)
{
	FString SaveDirectory = FPaths::ProjectSavedDir();
	FString FilePath = FPaths::Combine(*SaveDirectory, TEXT("MyFile.txt"));
	
	FFileHelper::SaveStringArrayToFile(Data, *FilePath);
}

